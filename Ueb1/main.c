#include <stdio.h>
#include <sys/time.h>   // for gettimeofday()
#include <unistd.h>      // for sleep()
#include <sys/sysinfo.h>// for systeminfo
#include <errno.h>      // for Errornumbers
#include <math.h>       // for Matrixmultiplikation
#include <stdlib.h>
#include <omp.h>        // for multicore process

const int ERROR = -1;
int max_threads = 0;

struct times {
    double sum;
    double sequentialCalculation;
    double parallelCalculation;
};

unsigned int getDimension() {
    struct sysinfo info;
    if (sysinfo(&info) == ERROR) {
        switch (errno) {
            case EACCES :
                printf("Access denied\n");
                return ERROR;
            default :
                printf("Some error occured\n");
                return ERROR;
        }
    }

    unsigned long long totalRam = info.mem_unit * info.totalram;
    unsigned long long freeRam = info.mem_unit * info.freeram;

    //Subtract 15% from available RAM to prevent it from overflowing.
    //DO NOT REMOVE IT OTHERWISE YOUR PC WILL CRASH WITH A PROBABILITY (happened twice to me)
    freeRam = (unsigned long long) ((double) freeRam * 0.85);

    unsigned long long maxSize = freeRam / sizeof(float);

    printf("Gesamtspeicher           : %11llu bytes\n", totalRam);
    printf("Freier Speicher          : %11llu bytes\n", freeRam);
    printf("Anzahl möglicher Floats  : %11llu floats\n", maxSize);
    //Each Vector needs space, 2 for the nxn-Matrix, one for the input vector and one for the output vector
    //Therefore use pq-formular because of : n^2+2n=MaximalVerfuegbareFloats

    unsigned int vectorSize = (int) pow((double) maxSize - 0.75F, 1.0F / 2.0F) - 1;
    printf("Maximal nutzbare Floats  : %11ld floats\n", (long) pow((double) vectorSize, 2) + 2 * vectorSize);
    printf("Maximale Vektor Größe    : %11d indices\n", vectorSize);
    printf("Anzahl Threads verfügbar : %d Threads\n", max_threads);
    printf("---------------------------------------------------\n");
    return vectorSize;
}

double calculateTimeDifference(struct timeval start, struct timeval end) {
    long seconds =  (end.tv_sec - start.tv_sec);
    int micros = abs((int)end.tv_usec - (int)start.tv_usec);
    return (double) seconds + (double) micros / 1e6f;
}

float* calculation(float** matrix, const float* vector, unsigned long dimension, struct times* times, unsigned int useParallel) {
    float* result = malloc(dimension * sizeof(float));

    struct timeval start, end;

    if (useParallel < 1) {
        omp_set_num_threads(max_threads);
    } else {
        omp_set_num_threads(1);
    }

    gettimeofday(&start, 0);

    #pragma omp parallel for
    for (unsigned long i = 0; i < dimension; i++) {
        for (unsigned long j = 0; j < dimension; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    gettimeofday(&end, 0);
    if (useParallel < 1) {
        times->parallelCalculation = calculateTimeDifference(start, end);
    } else {
        times->sequentialCalculation = calculateTimeDifference(start, end);
    }
    return result;
}

void calculateMatrix(float** matrix ,float* vector, float* result, unsigned int dimension, struct times* times) {
    result = calculation(matrix, vector, dimension, times, 0);
    for (unsigned int i = 0; i < dimension; i++) {
        result[i] = 0;
    }

    result = calculation(matrix, vector, dimension, times, 1);
    printf("\n");
    for (unsigned int i = 0; i < dimension; i++) {
        result[i] = 0;
    }
}

float** initMatrix(unsigned int dimension) {
    /*
		Generate 2 dimensional random TYPE matrix.
	*/
    float** matrix = malloc(dimension * sizeof(float*));

    #pragma omp parallel for
    for (unsigned int i = 0; i < dimension; i++) {
        matrix[i] = malloc(dimension * sizeof(float));
    }

    //Random seed
//    unsigned short* seed = (unsigned short *) time(0);
//    seed48(seed);
    #pragma omp parallel for
    for (unsigned int i = 0; i < dimension; i++) {
        for (unsigned int j = 0; j < dimension; j++) {
            matrix[i][j] = (float)  lrand48() / 100000.0f;
        }
    }
    return matrix;
}

float* initVector(unsigned int dimension) {
    float* vector = malloc(dimension * sizeof(float));

    #pragma omp parallel for
    for (unsigned int i = 0; i < dimension; i++) {
        vector[i] = (float)  lrand48() / 100000.0f;
    }
    return vector;
}

void calculateWithDimension(float** matrix, float* vector, float* result, unsigned int dimension) {
    struct times times;

    // start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    //calculate the matrix multiplication
    calculateMatrix(matrix, vector, result, dimension, &times);

    // end timer
    gettimeofday(&end, NULL);

    times.sum = calculateTimeDifference(start, end);

    //print time results
    printf("===============================================================\n");
    printf("time in seconds for dimension %ld\n", dimension);
    printf("Time for sequential calculation: %f\n", times.sequentialCalculation);
    printf("Time for parallel calculation  : %f\n", times.parallelCalculation);
    printf("-------------------------------------------\n");
    printf("Time sum                       : %f\n", times.sum);
    // speedup difference
    if(times.parallelCalculation < times.sequentialCalculation){
      double calculatedSpeedFactorSeq = times.sequentialCalculation / times.parallelCalculation; 
      printf("\nParallel Computing is -- %f -- times faster\n", calculatedSpeedFactorSeq); 
    }
    else if(times.sequentialCalculation < times.parallelCalculation){
        double calculatedSpeedFactorPar = times.parallelCalculation / times.sequentialCalculation; 
        printf("\nSequential Computing is -- %f -- times faster\n", calculatedSpeedFactorPar); 
    }
}

int main() {
    //gets maximum number of threads
    max_threads = omp_get_max_threads();

    //gets the maximum size of an array
    unsigned int max_dimension = getDimension();

    //measures time to initialize matrix and vectors
    struct timeval startInit, endInit;
    gettimeofday(&startInit, NULL);

    //initializes the array and vectors only once
    float** matrix = initMatrix(max_dimension);
    float* vector = initVector(max_dimension);
    float* result = malloc(max_dimension * sizeof(float));

    //stop time measure for the initialization
    gettimeofday(&endInit, NULL);

    double timeToInit = calculateTimeDifference(startInit, endInit);

    printf("Time to initialize: %f2\n", timeToInit);
    printf("===================================================\n");
    printf("\nStarting calculations:\n\n");
    //measure times for different matrix dimensions
    calculateWithDimension(matrix, vector, result, 100);
    calculateWithDimension(matrix, vector, result, 1000);
    calculateWithDimension(matrix, vector, result, 10000);
    calculateWithDimension(matrix, vector, result, 20000);
    calculateWithDimension(matrix, vector, result, 30000);
    calculateWithDimension(matrix, vector, result, max_dimension);

    //free memory
    for (unsigned long i = 0; i < max_dimension; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(vector);
    free(result);
    return 0;
}