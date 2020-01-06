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

void exercise1();
void exercise3();

int main() {
	//init random seed
	srand((unsigned) second());

	//exercise1();
	exercise3();
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


void exercise1() {
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

	cudaFree(matrix);
	cudaCheckError();

	cudaFree(interimResultsA);
	cudaCheckError();

	cudaFree(interimResultsB);
	cudaCheckError();
}

//=====================================================================================
//									Aufgabe 2										 //
//=====================================================================================

/*
	NVIDIA RTX 2080 (8GB Memory)
	Every calculation was performed 10 times and the average is found in the table below


	Dimension |	CPU				| GPU			 | factor
	-----------------------------------------------------
	10		  | 0.000001		| 0.000168		 | 0,006
	100		  | 0.000012		| 0.000311		 | 0,038
	1.000	  | 0.001100		| 0.002683		 | 0,409
	10.000	  | 0.106819		| 0.141655       | 0,75
	20.000	  | 0.422229		| 0.520808		 | 0,81
	30.000	  | 0.955817	    | 1.192795		 | 0,80

	As we can see in the table is the CPU always fast in this specifi calculation than the GPU
	But as the number of calculations get higher the CPU and GPU get closer with the time it takes to calculate. So its possible that the GPU is faster than the CPU with a bigger matrix. But due to memory limitations i cant test this.
*/

const int DIMENSION = 30000;
const int NUMBER_OF_BLOCKS = 100;
const int NUMBER_OF_THREADS = DIMENSION / NUMBER_OF_BLOCKS;

__global__ void calculateMatrix(unsigned int dimension, int* matrix, int* vector, long long* result) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int size = dimension * (dimension + 1) / 2;
	long index = size - ((dimension - threadId) * (dimension - threadId + 1) / 2);

	for (unsigned int j = 0; j < dimension - threadId; j++) {
		result[threadId] += matrix[index + j] * vector[j];
	}
}

void exercise3() {
	printf("Start of Exercise 3!\n");

	//measures time to initialize matrix and vectors
	double timeStartInitMatrix, timeStopInitMatrix, timeToInitMatrix;
	double timeStartInitVector, timeStopInitVector, timeToInitVector;
	double timeStartCalculate, timeStopCalculate, timeToCalculate;

	//initializes the matrix
	int* matrix;

	int size = (DIMENSION * (DIMENSION + 1)) / 2;
	timeStartInitMatrix = second();

	cudaMallocManaged(&matrix, size * sizeof(int));
	cudaCheckError();

	for (int i = 0; i < size; i++) {
		matrix[i] = rand() % 50;
	}

	timeStopInitMatrix = second();
	timeToInitMatrix = timeStopInitMatrix - timeStartInitMatrix;
	printf("Time to init Matrix: %f\n", timeToInitMatrix);

	//initializes the vector
	int* vector;
	timeStartInitVector = second();

	cudaMallocManaged(&vector, DIMENSION * sizeof(int));
	cudaCheckError();

	for (int i = 0; i < DIMENSION; i++) {
		vector[i] = rand() % 50;
	}

	timeStopInitVector = second();
	timeToInitVector = timeStopInitVector - timeStartInitVector;
	printf("Time to init Vector: %f\n", timeToInitVector);

	//initialize result vector
	long long* result;

	cudaMallocManaged(&result, DIMENSION * sizeof(long long));
	cudaCheckError();

	//calculate result
	timeStartCalculate = second();
	calculateMatrix << <NUMBER_OF_BLOCKS, NUMBER_OF_THREADS >> > (DIMENSION, matrix, vector, result);
	cudaCheckError();
	cudaDeviceSynchronize();
	timeStopCalculate = second();
	timeToCalculate = timeStopCalculate - timeStartCalculate;
	printf("Time to Calculate: %f\n", timeToCalculate);

	cudaFree(matrix);
	cudaFree(vector);
	cudaFree(result);
}

//=====================================================================================
//									Aufgabe 5										 //
//=====================================================================================
