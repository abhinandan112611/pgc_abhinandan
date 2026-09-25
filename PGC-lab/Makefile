# Master Makefile for Parallel & Grid Computing Laboratory (PGC-Lab)
CC = gcc
MPICC = mpicc

CFLAGS = -O2 -Wall
OMPFLAGS = -fopenmp

all: sequential openmp mpi

sequential: 01-Sequential/matrix_sequential.c
	$(CC) $(CFLAGS) 01-Sequential/matrix_sequential.c -o matrix_sequential

openmp: 02-OpenMP/matrix_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) 02-OpenMP/matrix_openmp.c -o matrix_openmp

mpi: 03-MPI/matrix_mpi.c
	@which $(MPICC) > /dev/null 2>&1 && $(MPICC) $(CFLAGS) 03-MPI/matrix_mpi.c -o matrix_mpi || echo "MPI (mpicc) not found, skipping MPI build."

clean:
	rm -f matrix_sequential matrix_openmp matrix_mpi

.PHONY: all sequential openmp mpi clean
