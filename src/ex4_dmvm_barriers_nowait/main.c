/**
 * TP4 - Exercise 4: Synchronization and Barrier Cost (DMVM)
 * 
 * Dense Matrix-Vector Multiplication (DMVM): y = A * x
 * 
 * This program implements DMVM with three parallel versions to study
 * the impact of barriers and nowait on performance:
 * 
 * Version 1: Implicit barrier (normal omp for without nowait)
 * Version 2: schedule(dynamic) with nowait
 * Version 3: schedule(static) with nowait
 * 
 * Metrics measured:
 * - CPU time
 * - Speedup (T_serial / T_parallel)
 * - Efficiency (Speedup / num_threads)
 * - MFLOP/s = (2*N*N) / (time * 10^6)  [2 FLOPs per element: multiply + add]
 * 
 * Author: Samia Lachgar
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Default matrix size */
#define DEFAULT_N 2000

/* Number of repetitions for timing */
#define REPETITIONS 5

/**
 * Allocate matrix with error checking
 */
double* allocate_matrix(int n) {
    double *mat = (double *)malloc(n * n * sizeof(double));
    if (mat == NULL) {
        fprintf(stderr, "Error: Failed to allocate %dx%d matrix\n", n, n);
        exit(EXIT_FAILURE);
    }
    return mat;
}

/**
 * Allocate vector with error checking
 */
double* allocate_vector(int n) {
    double *vec = (double *)malloc(n * sizeof(double));
    if (vec == NULL) {
        fprintf(stderr, "Error: Failed to allocate vector of size %d\n", n);
        exit(EXIT_FAILURE);
    }
    return vec;
}

/**
 * Initialize matrix with random values
 */
void init_matrix(double *A, int n) {
    srand(42);  /* Deterministic initialization */
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)(rand() % 100) / 100.0;
    }
}

/**
 * Initialize vector with random values
 */
void init_vector(double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (double)(rand() % 100) / 100.0;
    }
}

/**
 * Zero out vector
 */
void zero_vector(double *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
    }
}

/**
 * Serial DMVM: y = A * x
 */
void dmvm_serial(const double *A, const double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * Version 1: Parallel DMVM with implicit barrier
 * 
 * Standard OpenMP parallelization with implicit barrier at the end
 * of the parallel for loop. All threads synchronize before continuing.
 * 
 * Barrier overhead: Threads must wait for the slowest thread.
 */
void dmvm_v1_barrier(const double *A, const double *x, double *y, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
    /* Implicit barrier here - all threads wait */
}

/**
 * Version 2: Parallel DMVM with schedule(dynamic) and nowait
 * 
 * Dynamic scheduling: work is distributed in chunks at runtime.
 * nowait: threads don't wait at the end of the loop.
 * 
 * Dynamic scheduling helps with load imbalance but has higher overhead.
 * nowait allows threads to continue without synchronizing.
 * 
 * SAFETY NOTE: Using nowait is SAFE here because:
 * - Each row of y is written by exactly one iteration
 * - No iteration depends on results from other iterations
 * - After the parallel region, y is fully computed
 */
void dmvm_v2_dynamic_nowait(const double *A, const double *x, double *y, int n) {
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 64) nowait
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A[i * n + j] * x[j];
            }
            y[i] = sum;
        }
        /* No barrier - threads may exit at different times */
    }
    /* Implicit barrier at end of parallel region ensures completion */
}

/**
 * Version 3: Parallel DMVM with schedule(static) and nowait
 * 
 * Static scheduling: work is divided equally among threads at compile time.
 * nowait: threads don't wait at the end of the loop.
 * 
 * Static scheduling has lower overhead than dynamic.
 * Combined with nowait, this should be the fastest version.
 * 
 * SAFETY: Same reasoning as Version 2 - each y[i] written once.
 */
void dmvm_v3_static_nowait(const double *A, const double *x, double *y, int n) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A[i * n + j] * x[j];
            }
            y[i] = sum;
        }
        /* No barrier within parallel region */
    }
    /* Implicit barrier at end of parallel region */
}

/**
 * Validate results against serial version
 */
int validate_result(const double *y_ref, const double *y_test, int n, const char *version) {
    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(y_ref[i] - y_test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    if (max_diff > 1e-10) {
        printf("WARNING: %s max difference = %.2e\n", version, max_diff);
        return 0;
    }
    return 1;
}

/**
 * Run benchmarks and measure timing
 */
typedef struct {
    double time_sec;
    double speedup;
    double efficiency;
    double mflops;
} BenchmarkResult;

BenchmarkResult benchmark_version(
    void (*dmvm_func)(const double*, const double*, double*, int),
    const double *A, const double *x, double *y, int n,
    double serial_time, int num_threads
) {
    BenchmarkResult result;
    double times[REPETITIONS];
    
    /* Warm-up run */
    dmvm_func(A, x, y, n);
    
    /* Timed runs */
    for (int r = 0; r < REPETITIONS; r++) {
        zero_vector(y, n);
        double t_start = omp_get_wtime();
        dmvm_func(A, x, y, n);
        double t_end = omp_get_wtime();
        times[r] = t_end - t_start;
    }
    
    /* Use minimum time (best case without system interference) */
    result.time_sec = times[0];
    for (int r = 1; r < REPETITIONS; r++) {
        if (times[r] < result.time_sec) {
            result.time_sec = times[r];
        }
    }
    
    /* Compute metrics */
    result.speedup = serial_time / result.time_sec;
    result.efficiency = result.speedup / num_threads * 100.0;
    
    /* MFLOP/s: 2*N*N FLOPs (one multiply + one add per element) */
    double flops = 2.0 * n * n;
    result.mflops = flops / (result.time_sec * 1e6);
    
    return result;
}

/**
 * Print results table
 */
void print_results_table(int n, int threads, 
                        BenchmarkResult *serial,
                        BenchmarkResult *v1, 
                        BenchmarkResult *v2, 
                        BenchmarkResult *v3) {
    printf("\n");
    printf("=========================================================================\n");
    printf("                         BENCHMARK RESULTS\n");
    printf("=========================================================================\n");
    printf("Matrix size: %d x %d  |  Threads: %d  |  Repetitions: %d\n", 
           n, n, threads, REPETITIONS);
    printf("-------------------------------------------------------------------------\n");
    printf("%-30s %10s %10s %10s %12s\n", 
           "Version", "Time (s)", "Speedup", "Eff (%)", "MFLOP/s");
    printf("-------------------------------------------------------------------------\n");
    printf("%-30s %10.6f %10.2f %10.1f %12.2f\n",
           "Serial", serial->time_sec, 1.0, 100.0, serial->mflops);
    printf("%-30s %10.6f %10.2f %10.1f %12.2f\n",
           "V1: Implicit barrier", v1->time_sec, v1->speedup, v1->efficiency, v1->mflops);
    printf("%-30s %10.6f %10.2f %10.1f %12.2f\n",
           "V2: dynamic + nowait", v2->time_sec, v2->speedup, v2->efficiency, v2->mflops);
    printf("%-30s %10.6f %10.2f %10.1f %12.2f\n",
           "V3: static + nowait", v3->time_sec, v3->speedup, v3->efficiency, v3->mflops);
    printf("=========================================================================\n");
}

/**
 * Write results to CSV file
 */
void write_csv_header(FILE *fp) {
    fprintf(fp, "n,threads,version,time_sec,speedup,efficiency,mflops\n");
}

void write_csv_row(FILE *fp, int n, int threads, const char *version, BenchmarkResult *r) {
    fprintf(fp, "%d,%d,%s,%.6f,%.4f,%.2f,%.2f\n",
            n, threads, version, r->time_sec, r->speedup, r->efficiency, r->mflops);
}

void print_usage(const char *progname) {
    printf("Usage: %s [OPTIONS]\n", progname);
    printf("\nOptions:\n");
    printf("  -n SIZE       Matrix size (default: %d)\n", DEFAULT_N);
    printf("  -t THREADS    Number of threads (overrides OMP_NUM_THREADS)\n");
    printf("  -o FILE       Output CSV file\n");
    printf("  --scaling     Run thread scaling study (1,2,4,8,16 threads)\n");
    printf("  -h, --help    Show this help message\n");
}

int main(int argc, char *argv[]) {
    int n = DEFAULT_N;
    int num_threads = omp_get_max_threads();
    const char *output_file = NULL;
    int scaling_study = 0;
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
            omp_set_num_threads(num_threads);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--scaling") == 0) {
            scaling_study = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    
    printf("=========================================================================\n");
    printf("TP4 - Exercise 4: DMVM with Barriers and Nowait\n");
    printf("=========================================================================\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("FLOPs per DMVM: %.2e\n", 2.0 * n * n);
    printf("\n");
    
    /* Allocate memory */
    double *A = allocate_matrix(n);
    double *x = allocate_vector(n);
    double *y_serial = allocate_vector(n);
    double *y_v1 = allocate_vector(n);
    double *y_v2 = allocate_vector(n);
    double *y_v3 = allocate_vector(n);
    
    /* Initialize */
    printf("Initializing matrix and vector...\n");
    init_matrix(A, n);
    init_vector(x, n);
    
    /* Serial reference */
    printf("Running serial version for reference...\n");
    double t_serial_start = omp_get_wtime();
    for (int r = 0; r < REPETITIONS; r++) {
        dmvm_serial(A, x, y_serial, n);
    }
    double t_serial_end = omp_get_wtime();
    double serial_time = (t_serial_end - t_serial_start) / REPETITIONS;
    
    BenchmarkResult serial_result = {
        .time_sec = serial_time,
        .speedup = 1.0,
        .efficiency = 100.0,
        .mflops = 2.0 * n * n / (serial_time * 1e6)
    };
    
    FILE *csv_fp = NULL;
    if (output_file != NULL) {
        csv_fp = fopen(output_file, "w");
        if (csv_fp != NULL) {
            write_csv_header(csv_fp);
        }
    }
    
    if (scaling_study) {
        /* Thread scaling study */
        int thread_counts[] = {1, 2, 4, 8, 16};
        int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
        
        printf("\n=== Thread Scaling Study ===\n");
        
        for (int t = 0; t < num_tests; t++) {
            int threads = thread_counts[t];
            omp_set_num_threads(threads);
            
            printf("\nRunning with %d thread(s)...\n", threads);
            
            BenchmarkResult v1 = benchmark_version(dmvm_v1_barrier, A, x, y_v1, n, 
                                                   serial_time, threads);
            validate_result(y_serial, y_v1, n, "V1");
            
            BenchmarkResult v2 = benchmark_version(dmvm_v2_dynamic_nowait, A, x, y_v2, n,
                                                   serial_time, threads);
            validate_result(y_serial, y_v2, n, "V2");
            
            BenchmarkResult v3 = benchmark_version(dmvm_v3_static_nowait, A, x, y_v3, n,
                                                   serial_time, threads);
            validate_result(y_serial, y_v3, n, "V3");
            
            print_results_table(n, threads, &serial_result, &v1, &v2, &v3);
            
            if (csv_fp != NULL) {
                write_csv_row(csv_fp, n, threads, "serial", &serial_result);
                write_csv_row(csv_fp, n, threads, "v1_barrier", &v1);
                write_csv_row(csv_fp, n, threads, "v2_dynamic_nowait", &v2);
                write_csv_row(csv_fp, n, threads, "v3_static_nowait", &v3);
            }
        }
    } else {
        /* Single thread count benchmark */
        printf("Running benchmarks with %d threads...\n", num_threads);
        
        BenchmarkResult v1 = benchmark_version(dmvm_v1_barrier, A, x, y_v1, n,
                                               serial_time, num_threads);
        validate_result(y_serial, y_v1, n, "V1");
        
        BenchmarkResult v2 = benchmark_version(dmvm_v2_dynamic_nowait, A, x, y_v2, n,
                                               serial_time, num_threads);
        validate_result(y_serial, y_v2, n, "V2");
        
        BenchmarkResult v3 = benchmark_version(dmvm_v3_static_nowait, A, x, y_v3, n,
                                               serial_time, num_threads);
        validate_result(y_serial, y_v3, n, "V3");
        
        print_results_table(n, num_threads, &serial_result, &v1, &v2, &v3);
        
        if (csv_fp != NULL) {
            write_csv_row(csv_fp, n, num_threads, "serial", &serial_result);
            write_csv_row(csv_fp, n, num_threads, "v1_barrier", &v1);
            write_csv_row(csv_fp, n, num_threads, "v2_dynamic_nowait", &v2);
            write_csv_row(csv_fp, n, num_threads, "v3_static_nowait", &v3);
        }
    }
    
    if (csv_fp != NULL) {
        fclose(csv_fp);
        printf("\nResults written to %s\n", output_file);
    }
    
    /* Discussion */
    printf("\n=========================================================================\n");
    printf("                         DISCUSSION\n");
    printf("=========================================================================\n");
    printf("\n1. WHY BARRIERS LIMIT SCALABILITY:\n");
    printf("   - Implicit barriers force all threads to synchronize\n");
    printf("   - Fastest threads wait for slowest thread to complete\n");
    printf("   - With load imbalance, barrier time increases\n");
    printf("   - Amdahl's Law: serial synchronization limits speedup\n");
    printf("\n2. WHEN NOWAIT BECOMES DANGEROUS:\n");
    printf("   - When subsequent code depends on loop results\n");
    printf("   - When threads access shared data written by other threads\n");
    printf("   - Data races if write-after-read or read-after-write conflicts\n");
    printf("   SAFE in this case: each y[i] written by exactly one iteration\n");
    printf("\n3. SCHEDULING COMPARISON:\n");
    printf("   - static: Low overhead, good for regular workloads\n");
    printf("   - dynamic: Higher overhead, better load balance\n");
    printf("   - For DMVM with uniform row computation, static is preferred\n");
    printf("=========================================================================\n");
    
    /* Cleanup */
    free(A);
    free(x);
    free(y_serial);
    free(y_v1);
    free(y_v2);
    free(y_v3);
    
    printf("\nExercise 4 completed successfully.\n");
    
    return EXIT_SUCCESS;
}
