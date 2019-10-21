#include <stdio.h>
#include <sys/time.h>   // for gettimeofday()
#include <unistd.h> 	  // for sleep()
#include <sys/sysinfo.h>// for systeminfo
#include <errno.h>      // for Errornumbers
#include <math.h>       // for Matrixmultiplikation
#include <omp.h>        // for multicore process

const int ERROR = -1;


unsigned int getMaxMatrixSize() 
{
  long maxSize;
    int ret;
    struct sysinfo info;
    if((ret = sysinfo(&info)) == ERROR){
        switch(errno){
            case EACCES : printf("Access denied\n");
                          return ERROR;
                          break;
            default : printf("Some error occured\n");
                      return ERROR;
        }
    }
    printf("Gesamtspeicher           : %11ld bytes\n", info.totalram);
    printf("Freier Speicher          : %11ld bytes\n", info.freeram);
    maxSize = info.freeram / sizeof(float);
    printf("Anzahl möglicher Floats  : %11ld floats\n", maxSize);
    //Each Vector needs space, 2 for the nxn-Matrix, one for the input vector and one for the output vector
    //Therefore use pq-formular because of : n^2+2n=MaximalVerfuegbareFloats
    unsigned int vectorSize = pow((double)maxSize-0.75F, 1.0F/2.0F)-1;
    printf("Maximal nutzbare Floats  : %11ld floats\n", (long)pow((double)vectorSize, 2)+2*vectorSize);
    printf("Maximale Vektor Größe    : %11d indices\n", vectorSize);
    printf("=================================\n");
    return vectorSize;
}

void berechneMatrix()
{

    printf("hier sollte die Matrix multiplikation berechnet werden");
}

void initMatrix(unsigned int maxMatrixSize){
  printf("die hier erstellte Matrix, muss eine %d x %d Matrix sein.", maxMatrixSize, maxMatrixSize);
}

int main()
{

  //init System max Size (RAM) nXn Matrix + 2 Vektoren (multiplikator und ergebnis)
  // SPEICHERFREIGABE NICHT VERGESSEN!!!!
  unsigned int maxMatrixSize = getMaxMatrixSize();
  // fülle Matrix und Vektor mit Random Float werten
  initMatrix(maxMatrixSize);


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