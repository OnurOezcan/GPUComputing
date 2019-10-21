#include <stdio.h>
#include <sys/time.h>   // for gettimeofday()
#include <unistd.h>      // for sleep()
#include <sys/sysinfo.h>// for systeminfo
#include <errno.h>      // for Errornumbers
#include <math.h>       // for Matrixmultiplikation
#include <stdlib.h>
//#include <zconf.h>
#include <omp.h>        // for multicore process

const int ERROR = -1;


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

    //Subtract 10% from available RAM to prevent it from overflowing.
    freeRam = (unsigned long long) ((double) freeRam * 0.9);

    unsigned long long maxSize = freeRam / sizeof(float);

    printf("Gesamtspeicher           : %11llu bytes\n", totalRam);
    printf("Freier Speicher          : %11llu bytes\n", freeRam);
    printf("Anzahl möglicher Floats  : %11llu floats\n", maxSize);
    //Each Vector needs space, 2 for the nxn-Matrix, one for the input vector and one for the output vector
    //Therefore use pq-formular because of : n^2+2n=MaximalVerfuegbareFloats

    unsigned int vectorSize = (int) pow((double) maxSize - 0.75F, 1.0F / 2.0F) - 1;
    printf("Maximal nutzbare Floats  : %11ld floats\n", (long) pow((double) vectorSize, 2) + 2 * vectorSize);
    printf("Maximale Vektor Größe    : %11d indices\n", vectorSize);
    printf("Anzahl Threads verfügbar : %d Threads\n", omp_get_num_threads());
    printf("=================================\n");
    return vectorSize;
}

float* sequentialMatrixCalculation(float** matrix, float* vector, unsigned long dimension) {
    float* result = malloc(dimension * sizeof(float));

    struct timeval t0, t1;
    gettimeofday(&t0, 0);

    #pragma omp parallel for
    for (unsigned long i = 0; i < dimension; i++) {
        for (unsigned long j = 0; j < dimension; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    gettimeofday(&t1, 0);
    double elapsed = ((float) t1.tv_sec - t0.tv_sec) * 1.0f + ((float) t1.tv_usec - t0.tv_usec) / 1000000.0f;
    printf("Benötigte Zeit: %f", elapsed);
    return result;
}

void calculateMatrix(float** matrix ,float* vector, unsigned long dimension) {
    float* result = sequentialMatrixCalculation(matrix, vector, dimension);

    printf("\n");
//    for (unsigned long i = 0; i < dimension; i++) {
//        printf("%f\n", result[i]);
//    }
}

float** initMatrix(unsigned long maxMatrixSize) {
//    printf("die hier erstellte Matrix, muss eine %lu x %lu Matrix sein.", maxMatrixSize, maxMatrixSize);

    /*
		Generate 2 dimensional random TYPE matrix.
	*/
    float** matrix = malloc(maxMatrixSize * sizeof(float*));

    #pragma omp parallel for
    for (unsigned long i = 0; i < maxMatrixSize; i++) {
        matrix[i] = malloc(maxMatrixSize * sizeof(float));
    }

    //Random seed
    srandom(time(0) + clock() + random());
    #pragma omp parallel for
    for (unsigned long i = 0; i < maxMatrixSize; i++) {
        for (unsigned long j = 0; j < maxMatrixSize; j++) {
            matrix[i][j] = (float) (rand() % 1000000 + 10);
        }
    }
    return matrix;
}

float* initVector(unsigned long dimension) {
    float* vector = malloc(dimension * sizeof(float));

    #pragma omp parallel for
    for (unsigned long i = 0; i < dimension; i++) {
        vector[i] = (float) (rand() % 1000000 + 10);
    }
    return vector;
}

int main() {
    //init System max Size (RAM) n X n Matrix and 2 vectors (multiplier and result)

    // start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    //gets the dimension for the matrix abd the vectors
    unsigned long dimension = getDimension();

    //fills matrix with random float values
    float** matrix = initMatrix(dimension);

    //fills vector with random float values
    float* vector = initVector(dimension);

    //calculate the matrix multiplication
    calculateMatrix(matrix, vector, dimension);

    // end timer
    gettimeofday(&end, NULL);

    // calculate Op time
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * (long) 1e6) + end.tv_usec) - (start.tv_usec);

    // print op time
    printf("Time elpased is %ld seconds and %ld micros\n", seconds, micros);

    //free memory
    free(matrix);

    return 0;
}