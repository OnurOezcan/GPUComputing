
#include "cuda_runtime.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>

void init();

// Aufgabe 2
__global__ void vecafs(float* a, float* b, float* c, int n) {
	int idx;
	idx = blockIdx.x + threadIdx.x; //proj(blockIdx,threadIdx.x,...)
	if (idx < n) {
		// this can be replaced by the following calculation:
		c[idx] = a[idx] -((-(idx & 1)) | 1) * b[idx];

		// In order to better explain the calculation, we break it down into its individual steps
		// 1. Step (id & 1) The bitwise operation gives us a 0 for an even number and a 1 for an odd number.
		// 2. Step With the minus beofre the first calculation we simply convert the positive 1 into a -1
		// 3. Step At the moment we get a -1 or 0 as the result, but we want a -1 or 1. The bitwise OR is used for this. The 0 becomes 1 and the -1 remains unchanged.
		// 4. At the moment we get a 1 for positive numbers and a -1 for negative numbers, but what we want is exactly the opposite. For this reason, with the last minus the whole thing is turned around once again.
		// A big advantage of the calculation is that all operations are carried out bitwise and the value ranges of the vectors are not restricted. A disadvantage is the multiplication with 1 or -1 at the end with b[idx]. 
	}
}

// Aufgabe 1 a)
__global__ void kernelOddEvenProjection() {
	const int index = blockIdx.x * blockDim.x + ( threadIdx.x << 1 ) + (blockIdx.x & 1) * (-blockDim.x + 1);
	printf("Block ID: %d, Block DIM: %d, Thread ID: %d, ID: %d\n", blockIdx.x, blockDim.x, threadIdx.x, index);
}

// Aufgabe 1 b)
void oddEven() {
	const int N = 5;
	const int M = 8;
	int test[N][M] = { {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0} };

	for (int y = 0; y < N; y++) {
		for (int x = 0; x < M; x++) {
			//formel
			test[y][x] = (x - ((x >= M >> 1)* M >> 1))* (N << 1) + y * 2 + (x >= M >> 1);

			printf("%d ", test[y][x]);
		}
		printf("\n");
	}
}

__global__ void firstParallel()
{
	printf("Das hier lauft parallel.\n");
}

int main() {
	const int blockDim = 5;
	const int threadCount = 5;

	printf("Aufgabe 1 a):\n");
	kernelOddEvenProjection <<<blockDim, threadCount>>> ();
	cudaDeviceSynchronize();

	printf("\nAufgabe 1 b):\n");
	oddEven();
	printf("\nAufgabe 3:\n");
	init();
}

//=========================================================================//
//                              Aufgabe 3                                  //
//=========================================================================//
const unsigned int M = 5;
const unsigned int N = 4;

__device__ void PrefixSum(int* num, int N) {
	int i, sum, sumold;
	sum = sumold = 0;
	for (i = 0; i < N; i++) {
		sum += num[i];
		num[i] = sumold;
		sumold = sum;
	}
}

__global__ void compact(int *a, int *listex, int *listey) {
	extern __shared__ int num[];
	int idx = M * threadIdx.x;
	//init shared memory with 0
	__syncthreads();

	if (threadIdx.x == 0) {
		for (int i = 0; i < N; i++) {
			num[i] = 0;
		}
	}
	__syncthreads();

	for (unsigned int i = 0; i < M; i++) {
		printf("Thread ID: %d, %d\n", threadIdx.x, a[idx + i]);
		if (a[idx + i] == 16) {
			num[threadIdx.x]++;
		}
	}

	__syncthreads();
	
	if (threadIdx.x == 0) {
		PrefixSum(num, blockDim.x);
	}

	__syncthreads();

	int tmp = 0;
	for (unsigned int i = 0; i < M; i++) {
		if (a[idx + i] == 16) {
			listex[num[threadIdx.x] + tmp] = threadIdx.x;
			listey[num[threadIdx.x] + tmp] = i;
			tmp++;
		}
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		for (int i = 0; i < N * M; i++) {
			printf("%d, %d\n", listex[i], listey[i]);
		}
	}
	__syncthreads();
}

void init() {
	int* listX;
	int* listY;

	int* mandelBrot;
	cudaMallocManaged(&mandelBrot, M * N * sizeof(int));
	cudaMallocManaged(&listX, M * sizeof(int));
	cudaMallocManaged(&listY, M * sizeof(int));

	//simulated calculation of the mandelbrot 
	mandelBrot[0] = 0;
	mandelBrot[1] = 4;
	mandelBrot[2] = 16;
	mandelBrot[3] = 8;
	mandelBrot[4] = 13;

	mandelBrot[5] = 4;
	mandelBrot[6] = 16;
	mandelBrot[7] = 16;
	mandelBrot[8] = 3;
	mandelBrot[9] = 14;

	mandelBrot[10] = 9;
	mandelBrot[11] = 12;
	mandelBrot[12] = 16;
	mandelBrot[13] = 5;
	mandelBrot[14] = 16;

	mandelBrot[15] = 1;
	mandelBrot[16] = 7;
	mandelBrot[17] = 15;
	mandelBrot[18] = 16;
	mandelBrot[19] = 14;

	compact << <1, N, N * sizeof(int)>> > (mandelBrot, listX, listY);

	cudaFree(mandelBrot);
	cudaFree(listX);
	cudaFree(listY);
}
