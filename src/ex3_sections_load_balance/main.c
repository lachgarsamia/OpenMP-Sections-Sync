/**
 * TP4 - Exercise 3: Load Balancing with Parallel Sections
 * 
 * This program demonstrates load balancing strategies using OpenMP parallel
 * sections to handle heterogeneous workloads.
 * 
 * Workload simulation:
 * - Light task:    ~10ms of computation
 * - Moderate task: ~50ms of computation  
 * - Heavy task:    ~100ms of computation
 * 
 * Approaches implemented:
 * 1. Naive: One section per task (poor load balance)
 * 2. Optimized: Split heavy task into smaller chunks across more sections
 * 3. Task-based: Using OpenMP tasks for dynamic scheduling
 * 
 * Author: Samia Lachgar
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Workload parameters (iterations to simulate work) */
#define LIGHT_WORK     10000000   /* ~10ms */
#define MODERATE_WORK  50000000   /* ~50ms */
#define HEAVY_WORK    100000000   /* ~100ms */

/* Number of chunks to split heavy work for optimization */
#define HEAVY_CHUNKS   4

/**
 * Simulate computational work by performing dummy calculations
 * Returns a meaningless result to prevent compiler optimization
 */
double do_work(long iterations, const char *task_name, int verbose) {
    double result = 0.0;
    double t_start = omp_get_wtime();
    
    for (long i = 0; i < iterations; i++) {
        result += sin((double)i) * cos((double)i);
    }
    
    double t_end = omp_get_wtime();
    
    if (verbose) {
        printf("  [Thread %d] %s: %.3f ms (result=%.6f)\n",
               omp_get_thread_num(), task_name, 
               (t_end - t_start) * 1000, result);
    }
    
    return result;
}

/**
 * Light workload task
 */
double task_light(int verbose) {
    return do_work(LIGHT_WORK, "Light task", verbose);
}

/**
 * Moderate workload task
 */
double task_moderate(int verbose) {
    return do_work(MODERATE_WORK, "Moderate task", verbose);
}

/**
 * Heavy workload task
 */
double task_heavy(int verbose) {
    return do_work(HEAVY_WORK, "Heavy task", verbose);
}

/**
 * Heavy workload task - chunked version for better distribution
 */
double task_heavy_chunk(int chunk_id, int verbose) {
    char name[64];
    snprintf(name, sizeof(name), "Heavy chunk %d/%d", chunk_id + 1, HEAVY_CHUNKS);
    return do_work(HEAVY_WORK / HEAVY_CHUNKS, name, verbose);
}

/**
 * Serial execution: All tasks run sequentially
 */
double run_serial(int verbose) {
    double result = 0.0;
    double t_start = omp_get_wtime();
    
    if (verbose) printf("Running serial execution...\n");
    
    result += task_light(verbose);
    result += task_moderate(verbose);
    result += task_heavy(verbose);
    
    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;
    
    if (verbose) {
        printf("Serial total time: %.3f ms\n\n", elapsed * 1000);
    }
    
    return elapsed;
}

/**
 * Naive parallel sections: One section per task
 * 
 * Problem: If we have 3 threads, work distribution is:
 * - Thread 0: Light (10ms)
 * - Thread 1: Moderate (50ms)  
 * - Thread 2: Heavy (100ms)
 * 
 * Total time = max(10, 50, 100) = 100ms (limited by heavy task)
 * Ideal time = (10 + 50 + 100) / 3 = 53ms
 */
double run_naive_sections(int verbose) {
    double result = 0.0;
    double r_light = 0.0, r_moderate = 0.0, r_heavy = 0.0;
    double t_start = omp_get_wtime();
    
    if (verbose) {
        printf("Running naive parallel sections...\n");
        printf("Thread assignment: Light->T0, Moderate->T1, Heavy->T2\n");
    }
    
    #pragma omp parallel shared(r_light, r_moderate, r_heavy)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                r_light = task_light(verbose);
            }
            
            #pragma omp section
            {
                r_moderate = task_moderate(verbose);
            }
            
            #pragma omp section
            {
                r_heavy = task_heavy(verbose);
            }
        }
    }
    
    result = r_light + r_moderate + r_heavy;
    
    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;
    
    if (verbose) {
        printf("Naive sections total time: %.3f ms\n\n", elapsed * 1000);
    }
    
    return elapsed;
}

/**
 * Optimized parallel sections: Split heavy task into chunks
 * 
 * Strategy: Divide the heavy task (100ms) into HEAVY_CHUNKS smaller tasks
 * This allows better load distribution across available threads.
 * 
 * With 4 chunks of heavy task + light + moderate = 6 sections total
 * Better chance of balanced work distribution
 */
double run_optimized_sections(int verbose) {
    double result = 0.0;
    double r_light = 0.0, r_moderate = 0.0;
    double r_heavy[HEAVY_CHUNKS] = {0.0};
    double t_start = omp_get_wtime();
    
    if (verbose) {
        printf("Running optimized parallel sections...\n");
        printf("Heavy task split into %d chunks for better balance\n", HEAVY_CHUNKS);
    }
    
    #pragma omp parallel shared(r_light, r_moderate, r_heavy)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                r_light = task_light(verbose);
            }
            
            #pragma omp section
            {
                r_moderate = task_moderate(verbose);
            }
            
            #pragma omp section
            {
                r_heavy[0] = task_heavy_chunk(0, verbose);
            }
            
            #pragma omp section
            {
                r_heavy[1] = task_heavy_chunk(1, verbose);
            }
            
            #pragma omp section
            {
                r_heavy[2] = task_heavy_chunk(2, verbose);
            }
            
            #pragma omp section
            {
                r_heavy[3] = task_heavy_chunk(3, verbose);
            }
        }
    }
    
    result = r_light + r_moderate;
    for (int i = 0; i < HEAVY_CHUNKS; i++) {
        result += r_heavy[i];
    }
    
    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;
    
    if (verbose) {
        printf("Optimized sections total time: %.3f ms\n\n", elapsed * 1000);
    }
    
    return elapsed;
}

/**
 * Task-based execution: Using OpenMP tasks for dynamic scheduling
 * 
 * OpenMP tasks provide more flexible load balancing:
 * - Tasks are placed in a queue
 * - Idle threads pick up tasks dynamically
 * - Better for irregular workloads
 * 
 * Note: This approach uses #pragma omp task instead of sections.
 * It's included to demonstrate an alternative that better handles
 * load imbalance while satisfying the spirit of parallel work distribution.
 */
double run_task_based(int verbose) {
    double result = 0.0;
    double r_light = 0.0, r_moderate = 0.0;
    double r_heavy[HEAVY_CHUNKS] = {0.0};
    double t_start = omp_get_wtime();
    
    if (verbose) {
        printf("Running task-based execution...\n");
        printf("Using OpenMP tasks for dynamic load balancing\n");
    }
    
    #pragma omp parallel shared(r_light, r_moderate, r_heavy)
    {
        #pragma omp single
        {
            /* Create tasks - will be distributed dynamically */
            #pragma omp task shared(r_light)
            {
                r_light = task_light(verbose);
            }
            
            #pragma omp task shared(r_moderate)
            {
                r_moderate = task_moderate(verbose);
            }
            
            /* Split heavy work into multiple smaller tasks */
            for (int i = 0; i < HEAVY_CHUNKS; i++) {
                #pragma omp task shared(r_heavy) firstprivate(i)
                {
                    r_heavy[i] = task_heavy_chunk(i, verbose);
                }
            }
        }
        /* Implicit taskwait at end of single */
    }
    
    result = r_light + r_moderate;
    for (int i = 0; i < HEAVY_CHUNKS; i++) {
        result += r_heavy[i];
    }
    
    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;
    
    if (verbose) {
        printf("Task-based total time: %.3f ms\n\n", elapsed * 1000);
    }
    
    return elapsed;
}

/**
 * Print timing comparison table
 */
void print_comparison(double t_serial, double t_naive, double t_optimized, 
                     double t_tasks, int num_threads) {
    printf("=====================================================\n");
    printf("              TIMING COMPARISON\n");
    printf("=====================================================\n");
    printf("Number of threads: %d\n\n", num_threads);
    
    printf("%-20s %12s %10s %10s\n", "Method", "Time (ms)", "Speedup", "Efficiency");
    printf("-----------------------------------------------------\n");
    printf("%-20s %12.2f %10.2f %9.1f%%\n", 
           "Serial", t_serial * 1000, 1.0, 100.0);
    printf("%-20s %12.2f %10.2f %9.1f%%\n", 
           "Naive Sections", t_naive * 1000, 
           t_serial / t_naive, 
           (t_serial / t_naive) / num_threads * 100);
    printf("%-20s %12.2f %10.2f %9.1f%%\n", 
           "Optimized Sections", t_optimized * 1000, 
           t_serial / t_optimized,
           (t_serial / t_optimized) / num_threads * 100);
    printf("%-20s %12.2f %10.2f %9.1f%%\n", 
           "Task-based", t_tasks * 1000, 
           t_serial / t_tasks,
           (t_serial / t_tasks) / num_threads * 100);
    printf("-----------------------------------------------------\n");
    
    /* Theoretical analysis */
    double total_work_ms = (LIGHT_WORK + MODERATE_WORK + HEAVY_WORK) / 1e6 * 
                           (t_serial / ((LIGHT_WORK + MODERATE_WORK + HEAVY_WORK) / 1e6));
    double ideal_parallel = total_work_ms / num_threads;
    
    printf("\nTheoretical Analysis:\n");
    printf("  Total work: %.0f ms (Light=%.0f, Moderate=%.0f, Heavy=%.0f)\n",
           t_serial * 1000,
           (double)LIGHT_WORK / (LIGHT_WORK + MODERATE_WORK + HEAVY_WORK) * t_serial * 1000,
           (double)MODERATE_WORK / (LIGHT_WORK + MODERATE_WORK + HEAVY_WORK) * t_serial * 1000,
           (double)HEAVY_WORK / (LIGHT_WORK + MODERATE_WORK + HEAVY_WORK) * t_serial * 1000);
    printf("  Ideal parallel time (perfect balance): %.2f ms\n", t_serial * 1000 / num_threads);
    printf("  Naive bound (limited by heavy task): ~%.0f ms of serial heavy time\n",
           (double)HEAVY_WORK / (LIGHT_WORK + MODERATE_WORK + HEAVY_WORK) * t_serial * 1000);
    printf("=====================================================\n");
}

/**
 * Write results to CSV file
 */
void write_results_csv(const char *filename, double t_serial, double t_naive, 
                      double t_optimized, double t_tasks, int num_threads) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Warning: Could not write to %s\n", filename);
        return;
    }
    
    fprintf(fp, "method,threads,time_ms,speedup,efficiency\n");
    fprintf(fp, "serial,1,%.3f,1.00,100.0\n", t_serial * 1000);
    fprintf(fp, "naive_sections,%d,%.3f,%.2f,%.1f\n", 
            num_threads, t_naive * 1000, t_serial / t_naive,
            (t_serial / t_naive) / num_threads * 100);
    fprintf(fp, "optimized_sections,%d,%.3f,%.2f,%.1f\n",
            num_threads, t_optimized * 1000, t_serial / t_optimized,
            (t_serial / t_optimized) / num_threads * 100);
    fprintf(fp, "task_based,%d,%.3f,%.2f,%.1f\n",
            num_threads, t_tasks * 1000, t_serial / t_tasks,
            (t_serial / t_tasks) / num_threads * 100);
    
    fclose(fp);
    printf("\nResults written to %s\n", filename);
}

int main(int argc, char *argv[]) {
    int verbose = 1;
    const char *output_file = NULL;
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            verbose = 0;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    int num_threads = omp_get_max_threads();
    
    printf("=====================================================\n");
    printf("TP4 - Exercise 3: Load Balancing with Parallel Sections\n");
    printf("=====================================================\n");
    printf("Number of threads: %d\n", num_threads);
    printf("Workload simulation:\n");
    printf("  Light:    %10ld iterations\n", (long)LIGHT_WORK);
    printf("  Moderate: %10ld iterations\n", (long)MODERATE_WORK);
    printf("  Heavy:    %10ld iterations\n", (long)HEAVY_WORK);
    printf("=====================================================\n\n");
    
    /* Run all implementations */
    double t_serial = run_serial(verbose);
    double t_naive = run_naive_sections(verbose);
    double t_optimized = run_optimized_sections(verbose);
    double t_tasks = run_task_based(verbose);
    
    /* Print comparison table */
    print_comparison(t_serial, t_naive, t_optimized, t_tasks, num_threads);
    
    /* Write CSV if requested */
    if (output_file != NULL) {
        write_results_csv(output_file, t_serial, t_naive, t_optimized, 
                         t_tasks, num_threads);
    }
    
    printf("\nExplanation of results:\n");
    printf("1. Naive sections: Limited by the heaviest task. With 3 sections,\n");
    printf("   execution time ≈ max(light, moderate, heavy) = heavy time.\n");
    printf("2. Optimized sections: Splitting heavy task allows better distribution.\n");
    printf("   More sections mean more opportunities for parallel execution.\n");
    printf("3. Task-based: Dynamic scheduling provides best load balance for\n");
    printf("   irregular workloads. Idle threads steal work from the queue.\n");
    
    printf("\n=====================================================\n");
    printf("Exercise 3 completed successfully.\n");
    printf("=====================================================\n");
    
    return EXIT_SUCCESS;
}
