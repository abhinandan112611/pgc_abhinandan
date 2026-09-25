/**
 * @file matrix_sequential.c
 * @brief Experiment 1: Sequential Matrix Multiplication (Baseline O(N^3))
 * @author Sai Sriram (https://github.com/saisriram03)
 * @course Parallel & Grid Computing Laboratory (PGC-Lab)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

int main(void) {
    printf("=====================================================\n");
    printf("   PGC-LAB EXPERIMENT 1: SEQUENTIAL MATRIX MULTIPLY   \n");
    printf("=====================================================\n");
    printf("Allocating memory for %d x %d matrices (Size: %.2f MB each)...\n", 
           N, N, (N * N * sizeof(double)) / (1024.0 * 1024.0));

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)malloc((size_t)N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Error: Insufficient memory allocated.\n");
        return 1;
    }

    printf("Initializing %d x %d matrices...\n", N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* Standard Serial Triple-Loop Multiplication */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (double)(end.tv_sec - start.tv_sec) +
                     (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\nSequential Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Execution Time = %f seconds\n", elapsed);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);
    return 0;
}
