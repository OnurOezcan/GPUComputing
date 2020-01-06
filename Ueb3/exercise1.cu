#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <cuda.h>
#include <math.h>      
#include <stdio.h>
#include <math.h>
#include <time.h> 
#include <omp.h>
#include <cooperative_groups.h>

// A routine to give access to a high precision timer on most systems.
#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double second(void)
{
	LARGE_INTEGER t;
	static double oofreq;
	static int checkedForHighResTimer;
	static BOOL hasHighResTimer;

	if (!checkedForHighResTimer) {
		hasHighResTimer = QueryPerformanceFrequency(&t);
		oofreq = 1.0 / (double)t.QuadPart;
		checkedForHighResTimer = 1;
	}
	if (hasHighResTimer) {
		QueryPerformanceCounter(&t);
		return (double)t.QuadPart * oofreq;
	}
	else {
		return (double)GetTickCount() * 1.0e-3;
	}
}
#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
double second(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}
#else
#error unsupported platform
#endif

int exercise1();
int exercise2();

int main() {
	//exercise1();
	exercise2();
}


void cudaCheckError() {
	cudaError_t e = cudaGetLastError();                                 
	if (e != cudaSuccess) {
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
		exit(0); 
	}
}

//=====================================================================================
//									Aufgabe 1										 //
//=====================================================================================
const int N = pow(10, 8);
#define BLOCK_SIZE 5000;
#define TOTAL_NUMBER_OF_THREADS 20000;

__global__ void addValuesA(int* matrix, int* interimResults) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int blockSize = BLOCK_SIZE;
	const int totalNumberOfThreads = TOTAL_NUMBER_OF_THREADS;

	const int max = blockSize * (i + 1) - 1;
	int index = blockSize * i;
	long long added = 0;

	for (; index <= max; index++) {
		added += matrix[index];
	}
	interimResults[i] = added;
}

__global__ void addValuesB(int* matrix, int* interimResults) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int blockSize = BLOCK_SIZE;
	const int totalNumberOfThreads = TOTAL_NUMBER_OF_THREADS;

	long long added = 0;
	for (int index = 0; index < blockSize; index++) {
		added += matrix[i + index * totalNumberOfThreads];
	}
	interimResults[i] = added;
}


int exercise1() {
	/*
		A is slightly faster than B due to the memory management. This is because the data is always loaded in blocks. Since A calculates the data linearly, fewer data blocks have to be loaded by the CPU
	*/
	const int numberOfBlocks = 32;
	const int numberOfThreads = 625;
	const int totalNumberOfThreads = TOTAL_NUMBER_OF_THREADS;

	printf("Start of Exercise 1!\n");
	int* matrix;
	int* interimResultsA;
	int* interimResultsB;
	long long resultA = 0;
	long long resultB = 0;

	//allocate global memory
	cudaMallocManaged(&matrix, N * sizeof(int)); 
	cudaCheckError();

	cudaMallocManaged(&interimResultsA, totalNumberOfThreads * sizeof(int));
	cudaCheckError(); 
	
	cudaMallocManaged(&interimResultsB, totalNumberOfThreads * sizeof(int));
	cudaCheckError();

	double timeStart, timeStop, timeLinear;
	double timeStartA, timeStopA, timeParallelA;
	double timeStartB, timeStopB, timeParallelB;


	// init with values
	// rand was not initialized with srand on purpose. In this case we always fill the matrix with the same "random" values, therefore we can 
	// compare it better 
	long long added = 0;
	timeStart = second();
	for (unsigned int i = 0; i < N; i++) {
		matrix[i] = rand() % 10;
		added += matrix[i];
	}

	timeStop = second();

	printf("Linear Calculated Checksum: %d\n", added);
	timeLinear = timeStop - timeStart;
	printf("Time: %f\n\n", timeLinear);

	//b)
	timeStartB = second();
	addValuesB << <numberOfBlocks, numberOfThreads >> > (matrix, interimResultsB);
	cudaCheckError();
	cudaDeviceSynchronize();
	timeStopB = second();
	timeParallelB = timeStopB - timeStartB;

	for (int i = 0; i < totalNumberOfThreads; i++) {
		resultB += interimResultsB[i];
	}
	printf("Sum of b): %d\n", resultB);
	printf("Time A: %f\n\n", timeParallelB);

	//a)
	timeStartA = second();
	addValuesA << <numberOfBlocks, numberOfThreads >> > (matrix, interimResultsA);
	cudaCheckError();
	cudaDeviceSynchronize();
	timeStopA = second();
	timeParallelA = timeStopA - timeStartA;

	for (int i = 0; i < totalNumberOfThreads; i++) {
		resultA += interimResultsA[i];
	}
	printf("Sum of a): %d\n", resultA);
	printf("Time A: %f\n\n", timeParallelA);

	

	return 0;
}
