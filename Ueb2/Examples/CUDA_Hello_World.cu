#include <stdio.h>

void helloCPU()
{
  printf("Hallo, die CPU laesst gruessen.\n");
}

/*
 *  `__global__` signalisiert eine Funktion, 
 *  die auf der GPU(device) laufen soll.
 */

__global__ void helloGPU()
{
  printf("Hallo, die GPU laesst gruessen.\n");
}

int main()
{
  helloCPU();


  /*
   * Beim Hinzufügen der Konfig mit der <<<...>>> Syntax
   * wird diese Funktion als Kernel auf dem GPU gestartet.
   */

  helloGPU<<<1, 1>>>();

  /*
   * `cudaDeviceSynchronize` blockiert den CPU stream,
   *  bis alle GPU-Kernel abgeschlossen sind.
   */

  cudaDeviceSynchronize();
}
