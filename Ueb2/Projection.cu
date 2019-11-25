#include <stdio.h>

__global__ void setThreadIndex(int** deviceMatrix, long matrixSize){
  int i = blockIdx.x*blockDim.x + threadIdx.x; // index
  if(i < matrixSize){
    for(i;i < matrixSize; i++){
    }
  }
}

//Aufgabe 2
__global void vecads(float *a, float *b, float* c, int n){
  int idx;
  idx = proj(blockIdx,threadIdx.x,...);
  if(idx < n){
    for(int i=n; i < n;i++){
      for(; idx%2 == 0; i++){
        c[idx] = a[idx] + b[idx];
      }
      for(; idx%2 != 0; i++){
        c[idx] = a[idx] - b[idx];
      }
    }
  }
}

int main(void){
  int width = 4;
  int heigth = 4;
  int threadSize = 4;     // B = Threads
  long matrixSize = (long)width*heigth;
  long blockSize = matrixSize+(threadSize-1)/threadSize; 
  int **hostMatrix, **deviceMatrix;
  

  //allocate host memory
  hostMatrix = (int**)malloc(matrixSize*sizeof(int));

  //allocate device memory
  cudaMalloc(&deviceMatrix, matrixSize*sizeof(int)); 

  //copy memory from host to device
  cudaMemcpy(deviceMatrix, hostMatrix, matrixSize*sizeof(int), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  setThreadIndex<<<blockSize,threadSize>>>(deviceMatrix, matrixSize);

  //copy memory from device to host
  cudaMemcpy(hostMatrix, deviceMatrix, matrixSize*sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaFree(deviceMatrix);
  free(hostMatrix);
}
