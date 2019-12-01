#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

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

	kernelOddEvenProjection <<<blockDim, threadCount >>>();
	cudaDeviceSynchronize();
	
	oddEven();
}
