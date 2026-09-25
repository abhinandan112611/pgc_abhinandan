/**
 * @file matrix_openmp.c
 * @brief Experiment 2: Shared-Memory OpenMP Matrix Multiplication
 * @author Sai Sriram (https://github.com/saisriram03)
 * @course Parallel & Grid Computing Laboratory (PGC-Lab)
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main(void) {
    printf("=====================================================\n");
    printf("     PGC-LAB EXPERIMENT 2: OPENMP MATRIX MULTIPLY     \n");
    printf("=====================================================\n");

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)malloc((size_t)N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    int threads_used = 0;
    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        threads_used = omp_get_num_threads();

        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;

    printf("OpenMP Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Number of Threads Used = %d\n", threads_used);
    printf("Execution Time = %f seconds\n", elapsed);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);
    return 0;
}
