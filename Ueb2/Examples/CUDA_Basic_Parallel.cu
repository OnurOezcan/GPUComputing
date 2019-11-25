#include <stdio.h>

__global__ void firstParallel()
{
  printf("Das hier lauft parallel.\n");
}

int main()
{
  firstParallel<<<5, 5>>>();
  cudaDeviceSynchronize();
}
