#include <stdio.h>
#include <sys/time.h>   // for gettimeofday()
#include <unistd.h> 	  // for sleep()
#include <sys/sysinfo.h>// for systeminfo
#include <errno.h>      // for Errornumbers
#include <math.h>       // for Matrixmultiplikation
#include <omp.h>        // for multicore process

const int ERROR = -1;




void initMemory() 
{
  printf("hier sollte die Ram berechnet werden");
}

void berechneMatrix()
{

    printf("hier sollte die Matrix berechnet werden");
}


int main()
{

  //init System max Size (RAM) nXn Matrix + 2 Vektoren (multiplikator und ergebnis)
  // SPEICHERFREIGABE NICHT VERGESSEN!!!!
  initMemory();

  // start timer
	struct timeval start, end;

	gettimeofday(&start, NULL);

  //Berechne die matrix Multiplikation
	berechneMatrix();

  // end timer
	gettimeofday(&end, NULL);

  // calculate Op time
	long seconds = (end.tv_sec - start.tv_sec);
	long micros = ((seconds * 1e6) + end.tv_usec) - (start.tv_usec);

  // print op time
	printf("Time elpased is %ld seconds and %ld micros\n", seconds, micros);

	return 0;
}