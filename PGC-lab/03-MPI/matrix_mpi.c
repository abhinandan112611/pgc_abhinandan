/**
 * @file matrix_mpi.c
 * @brief Experiment 3: Distributed-Memory MPI Matrix Multiplication
 * @author Sai Sriram (https://github.com/saisriram03)
 * @course Parallel & Grid Computing Laboratory (PGC-Lab)
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4000
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int main(int argc, char *argv[]) {
    int numtasks, taskid, numworkers, source, dest, mtype, rows, averow, extra, offset;
    double *A = NULL, *B = NULL, *C = NULL;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2) {
        if (taskid == MASTER) {
            printf("Error: Need at least 2 MPI processes to execute distributed benchmark.\n");
        }
        MPI_Finalize();
        return 1;
    }

    numworkers = numtasks - 1;

    /* Allocate Matrix B on all nodes (shared broadcast operand) */
    B = (double *)malloc((size_t)N * N * sizeof(double));

    /* ---------------- MASTER PROCESS ---------------- */
    if (taskid == MASTER) {
        printf("=====================================================\n");
        printf("      PGC-LAB EXPERIMENT 3: MPI DISTRIBUTED MULTIPLY  \n");
        printf("=====================================================\n");
        printf("Master process running with %d total MPI processes (%d workers)...\n", numtasks, numworkers);

        A = (double *)malloc((size_t)N * N * sizeof(double));
        C = (double *)malloc((size_t)N * N * sizeof(double));

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
        }

        double start_time = MPI_Wtime();

        /* Send Matrix B to all worker processes */
        averow = N / numworkers;
        extra = N % numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&A[offset * N], rows * N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(B, N * N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset += rows;
        }

        /* Receive completed slices from workers */
        mtype = FROM_WORKER;
        for (int i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset * N], rows * N, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
        }

        double end_time = MPI_Wtime();
        printf("\nMPI Distributed Matrix Multiplication Completed\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Number of MPI Processes = %d\n", numtasks);
        printf("Execution Time = %f seconds\n", end_time - start_time);
        printf("Verification C[0][0] = %.2f\n", C[0]);

        free(A);
        free(C);
    }

    /* ---------------- WORKER PROCESSES ---------------- */
    if (taskid > MASTER) {
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        double *sub_A = (double *)malloc((size_t)rows * N * sizeof(double));
        double *sub_C = (double *)malloc((size_t)rows * N * sizeof(double));

        MPI_Recv(sub_A, rows * N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(B, N * N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        /* Compute distributed matrix slice */
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += sub_A[i * N + k] * B[k * N + j];
                }
                sub_C[i * N + j] = sum;
            }
        }

        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(sub_C, rows * N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

        free(sub_A);
        free(sub_C);
    }

    free(B);
    MPI_Finalize();
    return 0;
}
