/**
 * TP4 - Exercise 1: Work Distribution with Parallel Sections
 * 
 * This program demonstrates the use of OpenMP parallel sections to compute
 * statistics (sum, max, standard deviation) on an array in parallel.
 * 
 * Key OpenMP features:
 * - #pragma omp parallel
 * - #pragma omp sections / section
 * - #pragma omp barrier (implicit and explicit)
 * - Shared variables with proper synchronization
 * 
 * Critical constraint: Section 3 (stddev) must use the sum computed in Section 1,
 * not recompute it. This requires proper synchronization.
 * 
 * Author: Samia Lachgar
 * Date: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000000  /* Array size */

/**
 * Initialize array with random values in range [0, 100)
 * Uses srand(0) for deterministic/reproducible results
 */
void initialize_array(double *arr, int size) {
    srand(0);  /* Deterministic initialization as per requirements */
    for (int i = 0; i < size; i++) {
        arr[i] = (double)(rand() % 100);
    }
}

/**
 * Compute sum of array elements (Section 1)
 */
double compute_sum(const double *arr, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

/**
 * Find maximum element in array (Section 2)
 */
double compute_max(const double *arr, int size) {
    double max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

/**
 * Compute standard deviation using pre-computed mean (Section 3)
 * stddev = sqrt(sum((x_i - mean)^2) / N)
 */
double compute_stddev(const double *arr, int size, double mean) {
    double variance = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = arr[i] - mean;
        variance += diff * diff;
    }
    return sqrt(variance / size);
}

/**
 * Serial version for validation and comparison
 */
void compute_serial(const double *arr, int size, double *sum, double *max_val, double *stddev) {
    double t_start = omp_get_wtime();
    
    *sum = compute_sum(arr, size);
    *max_val = compute_max(arr, size);
    double mean = *sum / size;
    *stddev = compute_stddev(arr, size, mean);
    
    double t_end = omp_get_wtime();
    printf("Serial execution time: %.6f seconds\n", t_end - t_start);
}

/**
 * Parallel version using OpenMP sections
 * 
 * Design notes:
 * - Section 1 computes sum
 * - Section 2 computes max (independent of sum)
 * - Section 3 computes stddev (depends on sum from Section 1)
 * 
 * Synchronization strategy:
 * - We use a two-phase approach:
 *   Phase 1: Compute sum and max in parallel (sections)
 *   Phase 2: After implicit barrier, compute stddev using the sum
 * 
 * This ensures Section 3 has access to the final sum value.
 */
void compute_parallel(const double *arr, int size, double *sum, double *max_val, double *stddev) {
    double t_start = omp_get_wtime();
    
    /* Shared variables for results */
    double local_sum = 0.0;
    double local_max = arr[0];
    double local_stddev = 0.0;
    double mean = 0.0;
    
    /* Flag to indicate sum is ready */
    int sum_ready = 0;
    
    #pragma omp parallel shared(local_sum, local_max, local_stddev, mean, sum_ready, arr, size)
    {
        /*
         * Phase 1: Compute sum and max in parallel using sections
         * The implicit barrier at the end of sections ensures both complete
         * before we proceed to stddev computation.
         */
        #pragma omp sections
        {
            /* Section 1: Compute sum */
            #pragma omp section
            {
                double section_sum = 0.0;
                for (int i = 0; i < size; i++) {
                    section_sum += arr[i];
                }
                local_sum = section_sum;
                printf("Thread %d: Computed sum = %.2f\n", omp_get_thread_num(), local_sum);
            }
            
            /* Section 2: Compute max (independent of sum) */
            #pragma omp section
            {
                double section_max = arr[0];
                for (int i = 1; i < size; i++) {
                    if (arr[i] > section_max) {
                        section_max = arr[i];
                    }
                }
                local_max = section_max;
                printf("Thread %d: Computed max = %.2f\n", omp_get_thread_num(), local_max);
            }
        }
        /* Implicit barrier here ensures sum and max are complete */
        
        /*
         * Phase 2: Compute mean and stddev
         * Only one thread computes mean, then all can use it for stddev
         */
        #pragma omp single
        {
            mean = local_sum / size;
            printf("Thread %d: Computed mean = %.2f\n", omp_get_thread_num(), mean);
        }
        /* Implicit barrier after single */
        
        /*
         * Section 3: Compute standard deviation
         * Uses the sum computed in Section 1 (via mean)
         * This satisfies the requirement: "Do not recompute the sum inside Section 3"
         */
        #pragma omp single
        {
            double variance = 0.0;
            for (int i = 0; i < size; i++) {
                double diff = arr[i] - mean;
                variance += diff * diff;
            }
            local_stddev = sqrt(variance / size);
            printf("Thread %d: Computed stddev = %.6f\n", omp_get_thread_num(), local_stddev);
        }
    }
    
    /* Copy results to output parameters */
    *sum = local_sum;
    *max_val = local_max;
    *stddev = local_stddev;
    
    double t_end = omp_get_wtime();
    printf("Parallel execution time: %.6f seconds\n", t_end - t_start);
}

/**
 * Alternative parallel version: All three computations in sections
 * with explicit synchronization for the dependency
 */
void compute_parallel_v2(const double *arr, int size, double *sum, double *max_val, double *stddev) {
    double t_start = omp_get_wtime();
    
    double local_sum = 0.0;
    double local_max = arr[0];
    double local_stddev = 0.0;
    volatile int sum_computed = 0;  /* Flag for synchronization */
    
    #pragma omp parallel shared(local_sum, local_max, local_stddev, sum_computed, arr, size)
    {
        #pragma omp sections
        {
            /* Section 1: Compute sum and set flag when done */
            #pragma omp section
            {
                double section_sum = 0.0;
                for (int i = 0; i < size; i++) {
                    section_sum += arr[i];
                }
                #pragma omp atomic write
                local_sum = section_sum;
                
                #pragma omp atomic write
                sum_computed = 1;
                
                printf("[V2] Thread %d: Computed sum = %.2f\n", omp_get_thread_num(), section_sum);
            }
            
            /* Section 2: Compute max (independent) */
            #pragma omp section
            {
                double section_max = arr[0];
                for (int i = 1; i < size; i++) {
                    if (arr[i] > section_max) {
                        section_max = arr[i];
                    }
                }
                #pragma omp atomic write
                local_max = section_max;
                
                printf("[V2] Thread %d: Computed max = %.2f\n", omp_get_thread_num(), section_max);
            }
            
            /* Section 3: Compute stddev - waits for sum to be ready */
            #pragma omp section
            {
                /* Spin-wait for sum to be computed */
                int ready = 0;
                while (!ready) {
                    #pragma omp atomic read
                    ready = sum_computed;
                }
                
                /* Read the sum value */
                double current_sum;
                #pragma omp atomic read
                current_sum = local_sum;
                
                double mean = current_sum / size;
                double variance = 0.0;
                for (int i = 0; i < size; i++) {
                    double diff = arr[i] - mean;
                    variance += diff * diff;
                }
                double section_stddev = sqrt(variance / size);
                
                #pragma omp atomic write
                local_stddev = section_stddev;
                
                printf("[V2] Thread %d: Computed stddev = %.6f (using sum = %.2f)\n", 
                       omp_get_thread_num(), section_stddev, current_sum);
            }
        }
    }
    
    *sum = local_sum;
    *max_val = local_max;
    *stddev = local_stddev;
    
    double t_end = omp_get_wtime();
    printf("[V2] Parallel execution time: %.6f seconds\n", t_end - t_start);
}

int main(int argc, char *argv[]) {
    printf("===========================================\n");
    printf("TP4 - Exercise 1: Parallel Sections Stats\n");
    printf("===========================================\n");
    printf("Array size: %d elements\n", N);
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("\n");
    
    /* Allocate array with error checking */
    double *arr = (double *)malloc(N * sizeof(double));
    if (arr == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for array\n");
        return EXIT_FAILURE;
    }
    
    /* Initialize array with random values */
    printf("Initializing array with random values (srand(0))...\n");
    initialize_array(arr, N);
    printf("\n");
    
    /* Variables to store results */
    double sum_serial, max_serial, stddev_serial;
    double sum_parallel, max_parallel, stddev_parallel;
    double sum_parallel_v2, max_parallel_v2, stddev_parallel_v2;
    
    /* Serial computation for validation */
    printf("--- Serial Computation ---\n");
    compute_serial(arr, N, &sum_serial, &max_serial, &stddev_serial);
    printf("Sum: %.2f, Max: %.2f, StdDev: %.6f\n", sum_serial, max_serial, stddev_serial);
    printf("\n");
    
    /* Parallel computation (two-phase approach) */
    printf("--- Parallel Computation (Two-Phase) ---\n");
    compute_parallel(arr, N, &sum_parallel, &max_parallel, &stddev_parallel);
    printf("Sum: %.2f, Max: %.2f, StdDev: %.6f\n", sum_parallel, max_parallel, stddev_parallel);
    printf("\n");
    
    /* Parallel computation (all sections with sync) */
    printf("--- Parallel Computation (All Sections with Sync) ---\n");
    compute_parallel_v2(arr, N, &sum_parallel_v2, &max_parallel_v2, &stddev_parallel_v2);
    printf("Sum: %.2f, Max: %.2f, StdDev: %.6f\n", sum_parallel_v2, max_parallel_v2, stddev_parallel_v2);
    printf("\n");
    
    /* Validation */
    printf("--- Validation ---\n");
    double eps = 1e-6;
    int valid = 1;
    
    if (fabs(sum_serial - sum_parallel) > eps) {
        printf("ERROR: Sum mismatch (serial=%.2f, parallel=%.2f)\n", sum_serial, sum_parallel);
        valid = 0;
    }
    if (fabs(max_serial - max_parallel) > eps) {
        printf("ERROR: Max mismatch (serial=%.2f, parallel=%.2f)\n", max_serial, max_parallel);
        valid = 0;
    }
    if (fabs(stddev_serial - stddev_parallel) > eps) {
        printf("ERROR: StdDev mismatch (serial=%.6f, parallel=%.6f)\n", stddev_serial, stddev_parallel);
        valid = 0;
    }
    if (fabs(sum_serial - sum_parallel_v2) > eps) {
        printf("ERROR: Sum mismatch V2 (serial=%.2f, parallel_v2=%.2f)\n", sum_serial, sum_parallel_v2);
        valid = 0;
    }
    
    if (valid) {
        printf("All results match! Implementation is correct.\n");
    }
    
    /* Cleanup */
    free(arr);
    
    printf("\n===========================================\n");
    printf("Exercise 1 completed successfully.\n");
    printf("===========================================\n");
    
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
