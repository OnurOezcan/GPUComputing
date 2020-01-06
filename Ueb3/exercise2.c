#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

// A routine to give access to a high precision timer on most systems.
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
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

//=====================================================================================
//									Aufgabe 2										 //
//=====================================================================================
#define MAX_DIMENSION 10;
int max_threads = 0;

int exercise2();

int main() {
    exercise2();
}

int* initMatrix(unsigned int dimension, int* matrix) {
    unsigned int size = (dimension * (dimension + 1)) / 2;
    matrix = (int*)malloc(size * sizeof(int));

    #pragma omp parallel for
    for (unsigned int i = 0; i < size; i++) {
        matrix[i] = (rand() % 50);
    }
    return matrix;
}

int* initVector(unsigned int dimension, int* vector) {
    vector = (int*)malloc(dimension * sizeof(int));

    #pragma omp parallel for
    for (unsigned int i = 0; i < dimension; i++) {
        vector[i] = (rand() % 50);
    }

    return vector;
}

long long* calculateMatrix(unsigned int dimension, int* matrix, int* vector) {
    long long* result = (long long*)malloc(dimension * sizeof(long long));

    //#pragma omp parallel for
    for (unsigned int i = 0; i < dimension; i++) {
        result[i] = 0;
        const unsigned int size = dimension * (dimension + 1) / 2;
        long index = size - ((dimension - i) * (dimension - i + 1) / 2);
        for (unsigned int j = 0; j < dimension - i; j++) {
            result[i] += matrix[index + j] * vector[j];
        }
    }
    return result;
}

/*
    i7 8700k 12 Threads @5GHz

    Every calculation was performed 10 times and the average is found in the table below

	Dimension |	single threaded	| multi threaded | factor
	-----------------------------------------------------
	10		  | 0.000001		| 0.000001		 | 0
	100		  | 0.000012		| 0.000177		 | 0,06
	1.000	  | 0.001100		| 0.000403		 | 2,7
	10.000	  | 0.106819		| 0.025364       | 4,2
	60.000	  | 3.834200		| 0.914354       | 4,19

    with only 100 elements the single threaded method is faster but after this the multi threaded version is always
    at leas twice as fast
*/
int exercise2() {
    //init rand
    srand((unsigned)second());

    //gets maximum number of threads
    max_threads = omp_get_max_threads();

    //set dimension
    unsigned int maxDimension = MAX_DIMENSION;

    //measures time to initialize matrix and vectors
    double timeStartInitMatrix, timeStopInitMatrix, timeToInitMatrix;
    double timeStartInitVector, timeStopInitVector, timeToInitVector;
    double timeStartCalculate, timeStopCalculate, timeToCalculate;

    //initializes the matrix
    int* matrix;
    timeStartInitMatrix = second();
    matrix = initMatrix(maxDimension, matrix);
    timeStopInitMatrix = second();
    timeToInitMatrix = timeStopInitMatrix - timeStartInitMatrix;
    printf("Time to init Matrix: %f\n", timeToInitMatrix);
    /*int index = 0;
    for (int i = 0; i < maxDimension; i++) {
        for (int j = 0; j < maxDimension - i; j++) {
            printf("%d,", matrix[index]);
            index++;
        }
        printf("\n");
    }*/

    //initializes the vector
    int* vector;
    timeStartInitVector = second();
    vector = initVector(maxDimension, vector);
    timeStopInitVector = second();
    timeToInitVector = timeStopInitVector - timeStartInitVector;
    printf("Time to init Vector: %f\n", timeToInitVector);
    /*for (int i = 0; i < maxDimension; i++) {
        printf("%d,", vector[i]);
    }
    printf("\n");*/

    //initialize result vector
    long long* result = (long long*)malloc(maxDimension * sizeof(long long));

    timeStartCalculate = second();
    result = calculateMatrix(maxDimension, matrix, vector);
    timeStopCalculate = second();
    timeToCalculate = timeStopCalculate - timeStartCalculate;
    /*for (int i = 0; i < maxDimension; i++) {
        printf("%d,", result[i]);
    }*/

    /*const int e = maxDimension * (maxDimension + 1) / 2;
    for (int i = 0; i < maxDimension; i++) {
        for (int j = 0; j < maxDimension - i; j++) {
            int in = e - ((maxDimension - i) * (maxDimension - i + 1) / 2) + j;
            printf("%d,", in);
        }
        printf("\n");
    }*/
    printf("Time to Calculate: %f\n", timeToCalculate);
    free(matrix);
    free(vector);
    free(result);
    return 0;
}
