/**
 * TP4 - Exercise 2: Exclusive Execution — Master vs Single
 * 
 * This program demonstrates the difference between #pragma omp master and
 * #pragma omp single directives in OpenMP.
 * 
 * Key OpenMP features:
 * - #pragma omp master: Only master thread (thread 0) executes, NO implicit barrier
 * - #pragma omp single: One thread executes (any), HAS implicit barrier
 * - #pragma omp for reduction: Parallel sum computation
 * 
 * Tasks:
 * 1. Master thread initializes a matrix
 * 2. Single thread prints the matrix
 * 3. All threads compute the sum of all elements in parallel
 * 
 * Comparison: Execution time with and without OpenMP
 * 
 * Author: Samia Lachgar
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Matrix dimensions - small for printing, can increase for benchmarking */
#define ROWS_SMALL 5
#define COLS_SMALL 5
#define ROWS_LARGE 1000
#define COLS_LARGE 1000

/**
 * Initialize matrix with values (row * cols + col)
 */
void initialize_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (double)(i * cols + j);
        }
    }
}

/**
 * Print matrix (only for small matrices)
 */
void print_matrix(const double *matrix, int rows, int cols) {
    printf("Matrix (%dx%d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%6.1f", matrix[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

/**
 * Compute sum of all matrix elements (serial version)
 */
double compute_sum_serial(const double *matrix, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j];
        }
    }
    return sum;
}

/**
 * Serial version: All operations performed sequentially
 */
double run_serial(int rows, int cols, int verbose) {
    double t_start, t_end;
    double sum = 0.0;
    
    /* Allocate matrix */
    double *matrix = (double *)malloc(rows * cols * sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return -1.0;
    }
    
    t_start = omp_get_wtime();
    
    /* Step 1: Initialize matrix */
    initialize_matrix(matrix, rows, cols);
    
    /* Step 2: Print matrix (only if small and verbose) */
    if (verbose && rows <= 10 && cols <= 10) {
        print_matrix(matrix, rows, cols);
    }
    
    /* Step 3: Compute sum */
    sum = compute_sum_serial(matrix, rows, cols);
    
    t_end = omp_get_wtime();
    
    if (verbose) {
        printf("Serial sum: %.2f\n", sum);
        printf("Serial time: %.6f seconds\n", t_end - t_start);
    }
    
    free(matrix);
    return t_end - t_start;
}

/**
 * OpenMP version: Using master, single, and parallel for reduction
 */
double run_openmp(int rows, int cols, int verbose) {
    double t_start, t_end;
    double sum = 0.0;
    
    /* Allocate matrix */
    double *matrix = (double *)malloc(rows * cols * sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return -1.0;
    }
    
    t_start = omp_get_wtime();
    
    #pragma omp parallel shared(matrix, sum) 
    {
        /*
         * Step 1: Master thread initializes the matrix
         * 
         * Using #pragma omp master:
         * - Only thread 0 executes this block
         * - NO implicit barrier after master block
         * - Other threads skip this section entirely
         * 
         * IMPORTANT: We need a barrier after master to ensure matrix
         * is fully initialized before other threads access it.
         */
        #pragma omp master
        {
            if (verbose) {
                printf("Thread %d (master): Initializing matrix...\n", 
                       omp_get_thread_num());
            }
            initialize_matrix(matrix, rows, cols);
        }
        /* Explicit barrier needed because master has no implicit barrier */
        #pragma omp barrier
        
        /*
         * Step 2: Single thread prints the matrix
         * 
         * Using #pragma omp single:
         * - Any one thread executes this block (not necessarily thread 0)
         * - HAS implicit barrier after (unless nowait is specified)
         * - Guarantees only one thread performs the print
         */
        #pragma omp single
        {
            if (verbose && rows <= 10 && cols <= 10) {
                printf("Thread %d (single): Printing matrix...\n", 
                       omp_get_thread_num());
                print_matrix(matrix, rows, cols);
            }
        }
        /* Implicit barrier here ensures print completes before sum computation */
        
        /*
         * Step 3: All threads compute sum in parallel
         * 
         * Using #pragma omp for reduction(+:sum):
         * - Work is distributed among all threads
         * - Each thread has a private copy of sum
         * - At the end, all private copies are combined
         */
        #pragma omp for reduction(+:sum)
        for (int i = 0; i < rows * cols; i++) {
            sum += matrix[i];
        }
        /* Implicit barrier at end of for */
        
        #pragma omp single
        {
            if (verbose) {
                printf("Thread %d: Final sum = %.2f\n", 
                       omp_get_thread_num(), sum);
            }
        }
    }
    
    t_end = omp_get_wtime();
    
    if (verbose) {
        printf("OpenMP sum: %.2f\n", sum);
        printf("OpenMP time: %.6f seconds\n", t_end - t_start);
    }
    
    free(matrix);
    return t_end - t_start;
}

/**
 * Write timing results to CSV file
 */
void write_results_csv(const char *filename, double serial_time, double omp_time, 
                       int threads, int rows, int cols) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Warning: Could not open %s for writing\n", filename);
        return;
    }
    
    fprintf(fp, "mode,threads,rows,cols,time_sec,speedup\n");
    fprintf(fp, "serial,1,%d,%d,%.6f,1.00\n", rows, cols, serial_time);
    fprintf(fp, "openmp,%d,%d,%d,%.6f,%.2f\n", threads, rows, cols, omp_time, 
            serial_time / omp_time);
    
    fclose(fp);
    printf("\nResults written to %s\n", filename);
}

void print_usage(const char *progname) {
    printf("Usage: %s [OPTIONS]\n", progname);
    printf("\nOptions:\n");
    printf("  --serial       Run serial version only\n");
    printf("  --omp          Run OpenMP version only\n");
    printf("  --both         Run both versions (default)\n");
    printf("  --benchmark    Run with large matrix for benchmarking\n");
    printf("  --output FILE  Write results to CSV file\n");
    printf("  --help         Show this help message\n");
    printf("\nEnvironment:\n");
    printf("  OMP_NUM_THREADS  Set number of OpenMP threads\n");
}

int main(int argc, char *argv[]) {
    int run_serial_flag = 1;
    int run_omp_flag = 1;
    int benchmark_flag = 0;
    const char *output_file = NULL;
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--serial") == 0) {
            run_serial_flag = 1;
            run_omp_flag = 0;
        } else if (strcmp(argv[i], "--omp") == 0) {
            run_serial_flag = 0;
            run_omp_flag = 1;
        } else if (strcmp(argv[i], "--both") == 0) {
            run_serial_flag = 1;
            run_omp_flag = 1;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark_flag = 1;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    
    /* Determine matrix size */
    int rows = benchmark_flag ? ROWS_LARGE : ROWS_SMALL;
    int cols = benchmark_flag ? COLS_LARGE : COLS_SMALL;
    int verbose = !benchmark_flag;
    
    printf("================================================\n");
    printf("TP4 - Exercise 2: Master vs Single Execution\n");
    printf("================================================\n");
    printf("Matrix size: %d x %d\n", rows, cols);
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("\n");
    
    /* Explain the difference between master and single */
    printf("=== OpenMP Directive Comparison ===\n");
    printf("\n#pragma omp master:\n");
    printf("  - Only thread 0 (master) executes the block\n");
    printf("  - NO implicit barrier at the end\n");
    printf("  - Requires explicit barrier if synchronization needed\n");
    printf("\n#pragma omp single:\n");
    printf("  - Any one thread can execute (not necessarily thread 0)\n");
    printf("  - HAS implicit barrier at the end (unless nowait)\n");
    printf("  - More flexible for load balancing\n");
    printf("\n");
    
    double serial_time = 0.0;
    double omp_time = 0.0;
    
    /* Run serial version */
    if (run_serial_flag) {
        printf("=== Serial Execution ===\n");
        serial_time = run_serial(rows, cols, verbose);
        printf("\n");
    }
    
    /* Run OpenMP version */
    if (run_omp_flag) {
        printf("=== OpenMP Execution ===\n");
        omp_time = run_openmp(rows, cols, verbose);
        printf("\n");
    }
    
    /* Print comparison */
    if (run_serial_flag && run_omp_flag) {
        printf("=== Performance Comparison ===\n");
        printf("Serial time:  %.6f seconds\n", serial_time);
        printf("OpenMP time:  %.6f seconds\n", omp_time);
        printf("Speedup:      %.2fx\n", serial_time / omp_time);
        printf("Efficiency:   %.2f%%\n", 
               (serial_time / omp_time) / omp_get_max_threads() * 100);
        
        /* Validate results */
        double *matrix = (double *)malloc(rows * cols * sizeof(double));
        if (matrix != NULL) {
            initialize_matrix(matrix, rows, cols);
            double expected_sum = 0.0;
            for (int i = 0; i < rows * cols; i++) {
                expected_sum += matrix[i];
            }
            /* Sum of 0, 1, 2, ..., n-1 = n*(n-1)/2 */
            double formula_sum = (double)(rows * cols) * (rows * cols - 1) / 2.0;
            printf("\nValidation:\n");
            printf("  Computed sum: %.2f\n", expected_sum);
            printf("  Formula sum:  %.2f\n", formula_sum);
            printf("  Match: %s\n", 
                   (expected_sum == formula_sum) ? "YES" : "NO");
            free(matrix);
        }
        
        /* Write to CSV if requested */
        if (output_file != NULL) {
            write_results_csv(output_file, serial_time, omp_time, 
                            omp_get_max_threads(), rows, cols);
        }
    }
    
    printf("\n================================================\n");
    printf("Exercise 2 completed successfully.\n");
    printf("================================================\n");
    
    return EXIT_SUCCESS;
}
