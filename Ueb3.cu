
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>
#include <math.h>

const int N = pow(10, 8);
#define BLOCK_SIZE 5000;

__global__ void addValuesA(int *matrix, int *result) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int blockSize = BLOCK_SIZE;

	const int max = blockSize * (i + 1) - 1;
	int index = blockSize * i;
	long long added = 0;

	for (; index <= max; index++) {
		added += matrix[index];
	}
	result[i] = added;
}

__global__ void addValuesB(int *matrix, int *result) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int blockSize = BLOCK_SIZE;

	long long added = 0;
	for (int index = 0; index < blockSize; index++) {
		added += matrix[i + index * 20000];
	}
	result[i] = added;
}


int main()
{
	const int numberOfBlocks = N / 20000;
	int* matrix;
	int* resultA;
	int* resultB;

	//allocate global memory
	cudaMallocManaged(&matrix, N * sizeof(int));
	cudaMallocManaged(&resultA, 20000 * sizeof(int));
	cudaMallocManaged(&resultB, 20000 * sizeof(int));

	// init with values
	// rand was not initialized with srand on purpose. In this case we always fill the matrix with the same "random" values, therefore we can 
	// compare it better 
	long long added = 0;
	for (unsigned int i = 0; i < N; i++) {
		matrix[i] = rand() % 10;
		added += matrix[i];
	}
	
	printf("Pruefsumme: %d\n", added);
	addValuesA<<<100,  200>>> (matrix, resultA);
	cudaDeviceSynchronize();

	long long psA = 0;
	for (unsigned int i = 0; i < 20000; i++) {
		psA += resultA[i];
	}
	printf("Ergebnis a): %d\n", psA);

	addValuesB <<<100, 200>>> (matrix, resultB);
	cudaDeviceSynchronize();

	long long psB = 0;
	for (unsigned int i = 0; i < 20000; i++) {
		psB += resultB[i];
	}

	printf("Ergebnis b): %d\n", psB);

    return 0;
}
