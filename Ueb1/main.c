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
    double init;
    double sequentialCalculation;
    double parallelCalculation;
};

unsigned long getDimension() {
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
    printf("Anzahl Threads verfügbar : %d Threads\n", omp_get_max_threads());
    printf("=================================\n");
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

void calculateMatrix(float** matrix ,float* vector, unsigned long dimension, struct times* times) {
    float* result1 = calculation(matrix, vector, dimension, times, 0);
    free(result1);
    float* result2 = calculation(matrix, vector, dimension, times, 1);
    free(result2);
    printf("\n");
}

float** initMatrix(unsigned long dimension) {
//    printf("die hier erstellte Matrix, muss eine %lu x %lu Matrix sein.", maxMatrixSize, maxMatrixSize);

    /*
		Generate 2 dimensional random TYPE matrix.
	*/
    float** matrix = malloc(dimension * sizeof(float*));

    #pragma omp parallel for
    for (unsigned long i = 0; i < dimension; i++) {
        matrix[i] = malloc(dimension * sizeof(float));
    }

    //Random seed
//    unsigned short* seed = (unsigned short *) time(0);
//    seed48(seed);
    #pragma omp parallel for
    for (unsigned long i = 0; i < dimension; i++) {
        for (unsigned long j = 0; j < dimension; j++) {
            matrix[i][j] = (float)  lrand48() / 100000.0f;
        }
    }
    return matrix;
}

float* initVector(unsigned long dimension) {
    float* vector = malloc(dimension * sizeof(float));

    #pragma omp parallel for
    for (unsigned long i = 0; i < dimension; i++) {
        vector[i] = (float)  lrand48() / 100000.0f;
    }
    return vector;
}

void calculateWithDimension(long dimension) {
    omp_set_num_threads(max_threads);
    //init System max Size (RAM) n X n Matrix and 2 vectors (multiplier and result)
    struct times times;

    // start timer
    struct timeval start, end, startInit, endInit;
    gettimeofday(&start, NULL);

//    //gets the dimension for the matrix abd the vectors
//    unsigned long dimension = getDimension();

    //start timer for initialization
    gettimeofday(&startInit, NULL);

    //fills matrix with random float values
    float** matrix = initMatrix(dimension);

    //fills vector with random float values
    float* vector = initVector(dimension);

    //end timer for initialization
    gettimeofday(&endInit, NULL);

    //calculate time for initialization;
    times.init = calculateTimeDifference(startInit, endInit);

    //calculate the matrix multiplication
    calculateMatrix(matrix, vector, dimension, &times);

    // end timer
    gettimeofday(&end, NULL);

    times.sum = calculateTimeDifference(start, end);

    double s = times.sequentialCalculation / times.parallelCalculation;

    //print time results
    printf("===============================================================\n\n\n");
    printf("time in seconds for dimension %ld\n", dimension);
    printf("Time to init matrix and vector : %f\n", times.init);
    printf("Time for sequential calculation: %f %f\n", times.sequentialCalculation, s);
    printf("Time for parallel calculation  : %f\n", times.parallelCalculation);
    printf("-------------------------------------------\n");
    printf("Time sum                       : %f\n", times.sum);

    //free memory
    free(matrix);
    free(vector);
}

int main() {
    max_threads = omp_get_max_threads();
    calculateWithDimension(100);
    sleep(2);
    calculateWithDimension(1000);
    sleep(2);
    calculateWithDimension(10000);
    sleep(2);
    calculateWithDimension(20000);
    sleep(2);
    calculateWithDimension(30000);
    sleep(2);
    calculateWithDimension(40000);
    sleep(2);
    unsigned long dimension = getDimension();
    calculateWithDimension(dimension);
//    //init System max Size (RAM) n X n Matrix and 2 vectors (multiplier and result)
//    struct times times;
//
//    // start timer
//    struct timeval start, end, startInit, endInit;
//    gettimeofday(&start, NULL);
//
//    //gets the dimension for the matrix abd the vectors
//    unsigned long dimension = getDimension();
//
//    //start timer for initialization
//    gettimeofday(&startInit, NULL);
//
//    //fills matrix with random float values
//    float** matrix = initMatrix(dimension);
//
//    //fills vector with random float values
//    float* vector = initVector(dimension);
//
//    //end timer for initialization
//    gettimeofday(&endInit, NULL);
//
//    //calculate time for initialization;
//    times.init = calculateTimeDifference(startInit, endInit);
//
//    //calculate the matrix multiplication
//    calculateMatrix(matrix, vector, dimension, &times);
//
//    // end timer
//    gettimeofday(&end, NULL);
//
//    times.sum = calculateTimeDifference(start, end);
//
//    //print time results
//    printf("time in seconds\n");
//    printf("Time to init matrix and vector : %f\n", times.init);
//    printf("Time for sequential calculation: %f\n", times.sequentialCalculation);
//    printf("Time for parallel calculation  : %f\n", times.parallelCalculation);
//    printf("-------------------------------------------\n");
//    printf("Time sum                       : %f\n", times.sum);
//
//    //free memory
//    free(matrix);
//    free(vector);

    return 0;
}