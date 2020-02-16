#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thread>
#include <chrono>
#include <vector>
#include <time.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <filesystem>
#include "imageLoader.h"

#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>  

#ifdef _DEBUG
#define DEBUG_CLIENTBLOCK new( _CLIENT_BLOCK, __FILE__, __LINE__)
#else
#define DEBUG_CLIENTBLOCK
#endif // _DEBUG

#ifdef _DEBUG
#define new DEBUG_CLIENTBLOCK
#endif

//=============================================================================================================================
//                                                          Structs
//=============================================================================================================================
struct times_t {
    double cpuWithIndexing = 0;
    double ompWithIndexing = 0;

    double cpuWithInlineIndexing = 0;
    double ompWithInlineIndexing = 0;

    double cpuWithoutIndexing = 0;
    double ompWithoutIndexing = 0;

    double cpuOptimized = 0;
    double ompOptimized = 0;

    double cpuCombinedSteps = 0;
    double ompCombinedSteps = 0;

    double cpuCombinedStepsOptimized = 0;
    double ompCombinedStepsOptimized = 0;

    double cpuPow = 0;
    double ompPow = 0;

    double gpu = 0;
    double gpuOptimized = 0;
    double gpuSquare32 = 0;
    double gpuSquare64 = 0;
    double gpuSquare128 = 0;
};
 
 //=============================================================================================================================
//                                                        Constants
//=============================================================================================================================
const int GRIDSIZE32 = 32;
const int RUNS = 2;

//=============================================================================================================================
//                                                  Function Definitions
//=============================================================================================================================
void printCudaDeviceInformation(int maxAvaialbeCores);
times_t executeSobelOperator(char* image, int maxAvaialbeCores);

//cpu sobel functions
void separateStepSobelCpuWithIndexing(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores);
void separateStepSobelCpuWithInlineIndexing(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores);
void separateStepSobelCpuWithoutIndexing(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores);
void separateStepSobelCpuOptimizedCalculation(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores);
void combinedStepsSobelCpu(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores);
void combinedStepsSobelCpuOptimized(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores);
void combinedStepsSobelCpuPow(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores);

//=============================================================================================================================
//                                                      Global Variables
//=============================================================================================================================
int imageWidth = 0;


/**
* Index function to access a 1d array like a 2d array
*/
int getIndex(int x, int y) {
    return imageWidth * y + x;
};

//same as inline function
inline int getIndexInline(int x, int y) {
    return imageWidth * y + x;
};

void cudaCheckError() {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
        exit(0);
    }
}

void writeImageAndFreeData(char* imageName, char* appendTxt, imgData* image) {
    //Uncomment to save result images!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //writeImage(imageName, appendTxt, *image);
    delete[] image->pixels;
}

__global__ void sobel_gpu(const byte* image, byte* result, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        dx = (image[width * (y - 1) + (x - 1)]) + (-image[width * (y - 1) + (x + 1)]) +
            (2 * image[width * y + (x - 1)]) + (-2 * image[width * y + (x + 1)]) +
            (image[width * (y + 1) + (x - 1)]) + (-image[width * (y + 1) + (x + 1)]);

        dy = (image[(y - 1) * width + (x - 1)]) + (2 * image[(y - 1) * width + x]) + (image[(y - 1) * width + (x + 1)]) +
            (-image[(y + 1) * width + (x - 1)]) + (-2 * image[(y + 1) * width + x]) + (-image[(y + 1) * width + (x + 1)]);

        result[(width - 2) * (y - 1) + (x - 1)] = sqrt((dx * dx) + (dy * dy));
    }
}

__global__ void sobelGpuOptimized(const byte* image, byte* result, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    int tmpWidth = width * y;
    int tmp_m = tmpWidth - width;
    int tmp_p = tmpWidth + width;
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {

        dx = (image[tmp_m + (x - 1)]) + (-image[tmp_m + (x + 1)]) +
            (2 * image[tmpWidth + (x - 1)]) + (-2 * image[tmpWidth + (x + 1)]) +
            (image[tmp_p + (x - 1)]) + (-image[tmp_p + (x + 1)]);

        dy = (image[tmp_m + (x - 1)]) + (2 * image[tmp_m + x]) + (image[tmp_m + (x + 1)]) +
            (-image[tmp_p + (x - 1)]) + (-2 * image[tmp_p + x]) + (-image[tmp_p + (x + 1)]);

        result[(width - 2) * (y - 1) + (x - 1)] = sqrt((dx * dx) + (dy * dy));
    }
}

__global__ void sobelGpuSquare32(const byte* image, byte* result, const unsigned int width, const unsigned int height) {
    extern __shared__ byte cachedPixels[];
    
    int x = threadIdx.x + blockIdx.x * blockDim.x - blockIdx.x - blockIdx.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y - blockIdx.y - blockIdx.y;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    if (x < width && y < height) {
            cachedPixels[(threadY * 32) + threadX] = image[width * y + x];
    }
    __syncthreads();
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1 && threadX > 0 && threadY > 0 && threadX < GRIDSIZE32 - 1 && threadY < GRIDSIZE32 - 1) {
        float dx, dy;
        int tmpWidth = GRIDSIZE32 * threadY;
        int tmp_m = tmpWidth - GRIDSIZE32;
        int tmp_p = tmpWidth + GRIDSIZE32;

        dx = (cachedPixels[tmp_m + (threadX - 1)]) + (-cachedPixels[tmp_m + (threadX + 1)]) +
            (2 * cachedPixels[tmpWidth + (threadX - 1)]) + (-2 * cachedPixels[tmpWidth + (threadX + 1)]) +
            (cachedPixels[tmp_p + (threadX - 1)]) + (-cachedPixels[tmp_p + (threadX + 1)]);

        dy = (cachedPixels[tmp_m + (threadX - 1)]) + (2 * cachedPixels[tmp_m + threadX]) + (cachedPixels[tmp_m + (threadX + 1)]) +
            (-cachedPixels[tmp_p + (threadX - 1)]) + (-2 * cachedPixels[tmp_p + threadX]) + (-cachedPixels[tmp_p + (threadX + 1)]);

        result[(width - 2) * (y - 1) + (x - 1)] = sqrt((dx * dx) + (dy * dy));
    }
}

__global__ void sobelGpuSquare64(const byte* image, byte* result, const unsigned int width, const unsigned int height) {
    extern __shared__ byte cachedPixels[];

    int x = ((threadIdx.x + blockIdx.x * blockDim.x) << 1) - blockIdx.x - blockIdx.x;
    int y = ((threadIdx.y + blockIdx.y * blockDim.y) << 1) - blockIdx.y - blockIdx.y;
    int threadX = threadIdx.x << 1;
    int threadY = threadIdx.y << 1;

    if (x <= width - 2 && y <= height - 2) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                cachedPixels[((threadY + i) * 64) + threadX + j] = image[width * (y + i) + (x + j)];
            }
        }
    }

    x++;
    y++;
    threadX++;
    threadY++;

    __syncthreads();
    if (x > 0 && y > 0 && x < width - 2 && y < height - 2 && threadX > 0 && threadY > 0 && threadX < 64 - 2 && threadY < 64 - 2) {
        
        for (int i = 0; i < 2; i++) {
            float dx, dy;
            int tmpWidth = 64 * (threadY + i);
            int tmp_m = tmpWidth - 64;
            int tmp_p = tmpWidth + 64;
            for (int j = 0; j < 2; j++) {
                dx = (cachedPixels[tmp_m + (threadX - 1 + j)]) + (-cachedPixels[tmp_m + (threadX + 1 + j)]) +
                    (2 * cachedPixels[tmpWidth + (threadX - 1 + j)]) + (-2 * cachedPixels[tmpWidth + (threadX + 1 + j)]) +
                    (cachedPixels[tmp_p + (threadX - 1 + j)]) + (-cachedPixels[tmp_p + (threadX + 1 + j)]);


                dy = (cachedPixels[tmp_m + (threadX - 1 + j)]) + (2 * cachedPixels[tmp_m + threadX + j]) + (cachedPixels[tmp_m + (threadX + 1 + j)]) +
                    (-cachedPixels[tmp_p + (threadX - 1 + j)]) + (-2 * cachedPixels[tmp_p + threadX + j]) + (-cachedPixels[tmp_p + (threadX + 1 + j)]);


                result[(width - 2) * (y - 1 + i) + (x - 1 + j)] = sqrt((dx * dx) + (dy * dy));
            }
        }
    }
}

__global__ void sobelGpuSquare128(const byte* image, byte* result, const unsigned int width, const unsigned int height) {
    extern __shared__ byte cachedPixels[];

    int x = ((threadIdx.x + blockIdx.x * blockDim.x) << 2) - blockIdx.x - blockIdx.x;
    int y = ((threadIdx.y + blockIdx.y * blockDim.y) << 2) - blockIdx.x - blockIdx.x;
    int threadX = threadIdx.x << 2;
    int threadY = threadIdx.y << 2;


    if (x >= width - 3 && x < width) {
        int tmp = x;
        x = width - 4;
        threadX = threadX - (tmp - x);
        if (threadX < 0) {
            threadX = 0;
        }
    }

    if (y >= height - 3 && y < height) {
        int tmp = y;
        y = height - 4;
        threadY = threadY - (tmp - y);
        if (threadY < 0) {
            threadY = 0;
        }
    }

    if (x < width - 3 && y < height - 3) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (blockIdx.x == 32 && blockIdx.y == 68) {
                    //printf("X: %d, Y:, %d, TX: %d, TY: %d, Index: %d\n", x, y, threadX, threadY, ((threadY + i) * 128) + threadX + j);
                }
                cachedPixels[((threadY + i) * 128) + threadX + j] = image[width * (y + i) + (x + j)];
            }
        }
    }

    if (x >= width - 3 && x < width) {
        int tmp = x;
        x = width - 5;
        threadX = threadX - (tmp - x);
    }

    if (y >= height - 3 && y < height) {
        int tmp = y;
        y = height - 5;
        threadY = threadY - (tmp - y);
    }

    __syncthreads();

    if (x > 3 && y > 3 && x < width - 4 && y < height - 4 && threadX > 3 && threadY > 3 && threadX < 128 - 4 && threadY < 128 - 4) {
        //printf("X: %d, Y:, %d\n",x, y);
        
        for (int i = 0; i < 4; i++) {
            float dx, dy;
            int tmpWidth = 128 * (threadY + i);
            int tmp_m = tmpWidth - 128;
            int tmp_p = tmpWidth + 128;
            for (int j = 0; j < 4; j++) {
                dx = (cachedPixels[tmp_m + (threadX - 1 + j)]) + (-cachedPixels[tmp_m + (threadX + 1 + j)]) +
                    (2 * cachedPixels[tmpWidth + (threadX - 1 + j)]) + (-2 * cachedPixels[tmpWidth + (threadX + 1 + j)]) +
                    (cachedPixels[tmp_p + (threadX - 1 + j)]) + (-cachedPixels[tmp_p + (threadX + 1 + j)]);


                dy = (cachedPixels[tmp_m + (threadX - 1 + j)]) + (2 * cachedPixels[tmp_m + threadX + j]) + (cachedPixels[tmp_m + (threadX + 1 + j)]) +
                    (-cachedPixels[tmp_p + (threadX - 1 + j)]) + (-2 * cachedPixels[tmp_p + threadX + j]) + (-cachedPixels[tmp_p + (threadX + 1 + j)]);


                result[(width - 2) * (y - 1 + i) + (x - 1 + j)] = sqrt((dx * dx) + (dy * dy));
            }
        }
    }
}

int main(int argc, char* argv[]) {
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    //wraps the input arguments in a vector
    std::vector<char*> arguments(argv, argv + argc);

    //gets the max available number of cpu cores
    int maxAvaialbeCores = std::thread::hardware_concurrency();

    //Check if the user started the program with a valid number of arguments
    if (arguments.size() < 2) {
        printf("%s: Invalid number of command line arguments. Exiting program\n", argv[0]);
        printf("Usage: %s [image.png]", argv[0]);
        return 1;
    }

    //the first argument is no longer needed, therefore it gets removed
    arguments.erase(arguments.begin());

    //print device properties
    printCudaDeviceInformation(maxAvaialbeCores);

    //average times and results
    times_t averageTimes, results;

    //load image (currently only png supported)
    for (char* image : arguments) {
        printf("\n\n########################################################\n");
        printf("#  Starting image processing: %-25.25s#\n", image);
        printf("########################################################\n");

        for (int i = 1; i <= RUNS; i++) {
            printf("\n\n########################################################\n");
            printf("#                      Run: %-27d#\n", i);
            printf("########################################################\n");
            results = executeSobelOperator(image, maxAvaialbeCores);

            averageTimes.cpuWithIndexing += results.cpuWithIndexing;
            averageTimes.ompWithIndexing += results.ompWithIndexing;

            averageTimes.cpuWithInlineIndexing += results.cpuWithInlineIndexing;
            averageTimes.ompWithInlineIndexing += results.ompWithInlineIndexing;

            averageTimes.cpuWithoutIndexing += results.cpuWithoutIndexing;
            averageTimes.ompWithoutIndexing += results.ompWithoutIndexing;

            averageTimes.cpuOptimized += results.cpuOptimized;
            averageTimes.ompOptimized += results.ompOptimized;

            averageTimes.cpuCombinedSteps += results.cpuCombinedSteps;
            averageTimes.ompCombinedSteps += results.ompCombinedSteps;

            averageTimes.cpuCombinedStepsOptimized += results.cpuCombinedStepsOptimized;
            averageTimes.ompCombinedStepsOptimized += results.ompCombinedStepsOptimized;

            averageTimes.cpuPow += results.cpuPow;
            averageTimes.ompPow += results.ompPow;

            averageTimes.gpu += results.gpu;
            averageTimes.gpuOptimized += results.gpuOptimized;
            averageTimes.gpuSquare32 += results.gpuSquare32;
            averageTimes.gpuSquare64 += results.gpuSquare64;
            averageTimes.gpuSquare128 += results.gpuSquare128;
        }
    }
    printf("\n\n########################################################\n");
    printf("#                        Results                       #\n");
    printf("########################################################\n");

    printf("CPU with indexing            = %*.2f msec\n", 5, averageTimes.cpuWithIndexing / 1000 / RUNS);
    printf("OMP with indexing            = %*.2f msec\n\n", 5, averageTimes.ompWithIndexing / 1000 / RUNS);

    printf("CPU with inline indexing     = %*.2f msec\n", 5, averageTimes.cpuWithInlineIndexing / 1000 / RUNS);
    printf("OMP with inline indexing     = %*.2f msec\n\n", 5, averageTimes.ompWithInlineIndexing / 1000 / RUNS);

    printf("CPU without indexing         = %*.2f msec\n", 5, averageTimes.cpuWithoutIndexing / 1000 / RUNS);
    printf("OMP without indexing         = %*.2f msec\n\n", 5, averageTimes.ompWithoutIndexing / 1000 / RUNS);

    printf("CPU optimized calc           = %*.2f msec\n", 5, averageTimes.cpuOptimized / 1000 / RUNS);
    printf("OMP optimized calc           = %*.2f msec\n\n", 5, averageTimes.ompOptimized / 1000 / RUNS);

    printf("CPU combined steps           = %*.2f msec\n", 5, averageTimes.cpuCombinedSteps / 1000 / RUNS);
    printf("OMP combined steps           = %*.2f msec\n\n", 5, averageTimes.ompCombinedSteps / 1000 / RUNS);

    printf("CPU combined steps optimized = %*.2f msec\n", 5, averageTimes.cpuCombinedStepsOptimized / 1000 / RUNS);
    printf("OMP combined steps optimized = %*.2f msec\n\n", 5, averageTimes.ompCombinedStepsOptimized / 1000 / RUNS);

    printf("CPU combined steps pow       = %*.2f msec\n", 5, averageTimes.cpuPow / 1000 / RUNS);
    printf("OMP combined steps pow       = %*.2f msec\n\n", 5, averageTimes.ompPow / 1000 / RUNS);

    printf("GPU basic                    = %*.2f msec\n", 5, averageTimes.gpu / 1000 / RUNS);
    printf("GPU calculation optimized    = %*.2f msec\n", 5, averageTimes.gpuOptimized / 1000 / RUNS);
    printf("GPU square 32                = %*.2f msec\n", 5, averageTimes.gpuSquare32 / 1000 / RUNS);
    printf("GPU square 64                = %*.2f msec\n", 5, averageTimes.gpuSquare64 / 1000 / RUNS);
    printf("GPU square 128               = %*.2f msec\n", 5, averageTimes.gpuSquare128 / 1000 / RUNS);
    return 0;
}

/**
* Output information about the host (CPU) and divce (GPU)
*/
void printCudaDeviceInformation(int maxAvaialbeCores) {
    cudaDeviceProp cudaDeviceProperties;
    cudaGetDeviceProperties(&cudaDeviceProperties, 0);

    printf("########################################################\n");
    printf("#                 Device Information                   #\n");
    printf("########################################################\n");
    printf("CPU: %d Threads\n", std::thread::hardware_concurrency());
    printf("GPU: %s\n\
     CUDA Version %d.%d\n\
     %zd MB global Memory\n\
     %zd KB shared Memory per Block\n\
     %d CUDA cores\n",
        cudaDeviceProperties.name, cudaDeviceProperties.major, 
        cudaDeviceProperties.minor, cudaDeviceProperties.totalGlobalMem >> 20, 
        cudaDeviceProperties.sharedMemPerBlock >> 10, cudaDeviceProperties.multiProcessorCount);
}

/** 
 * The original image is extended by one pixel up, down, left and right. The newly added columns and rows are filled as follows: 
 * Example of an image with 3x3 pixels which is extended to 5x5 pixels
 *                  A   A   B   C   C
 *  A   B   C       A   A   B   C   C
 *  D   E   F   ->  D   D   E   F   F
 *  G   H   I       G   G   H   I   I
 *                  G   G   H   I   I
 *
 * The original borders are copied to the newly added border. The same is done for the corners
*/
void fillExpandedPicture(const byte* originalImage, byte* expandedImage, const unsigned int width, const unsigned int height) {
    //copy data from original image to the expanded image and fill the new added rows/columns
    imageWidth = width;
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            //cases for corners
            if (y == 0 && x == 0) {
                expandedImage[getIndex(x, y)] = originalImage[getIndex(x, y)];
            } else if (y == 0 && x == width - 1) {
                expandedImage[getIndex(x, y)] = originalImage[getIndex(x - 2, y)];
            } else if (y == height - 1 && x == 0) {
                expandedImage[getIndex(x, y)] = originalImage[(width - 2) * (y - 2) + x];
            } else if (y == height - 1 && x == width - 1) {
                expandedImage[getIndex(x, y)] = originalImage[(width - 2) * (y - 2) + (x - 2)];
            
            //cases for edges
            } else if (y == 0) {
                expandedImage[getIndex(x, y)] = originalImage[getIndex(x - 1, y)];
            } else if (y > 0 && x == 0) {
                expandedImage[getIndex(x, y)] = originalImage[(width - 2) * (y - 1) + x];
            } else if (y > 0 && x == width - 1) {
                expandedImage[getIndex(x, y)] = originalImage[(width - 2) * (y - 1) + (x - 2)];
            } else if (y == height - 1 && x > 0) {
                expandedImage[getIndex(x, y)] = originalImage[(width - 2) * (y - 2) + (x - 1)];
            }

            //fill in the normal image
            else {
                expandedImage[getIndex(x, y)] = originalImage[(width - 2) * (y - 1) + (x - 1)];
            }
        }
    }
}

times_t executeSobelOperator(char* image, int maxAvailableCores) {
    times_t times;
    std::chrono::system_clock::time_point start;
    
    //==========================================================
    // 1. Step: load image
    //==========================================================
    imgData originalImage = loadImage(image);

    //==========================================================
    // 2. Step: allocate create image that is two wider and two 
    //          heigher than the original
    //==========================================================
    imgData expandedImage(new byte[(originalImage.width + 2) * (originalImage.height + 2)], originalImage.width + 2, originalImage.height + 2);
    byte* expandedGpuImage;
    cudaMallocManaged((void**)&expandedGpuImage, (expandedImage.width * expandedImage.height) * sizeof(byte));
    cudaCheckError();

    //not necessary: the "fillExpandedPicture" function writes to every entry in the array. Setting every pixel to 0 before is redundant
    //memset(expandedImage.pixels, 0, (expandedImage.width * expandedImage.height));
    fillExpandedPicture(originalImage.pixels, expandedImage.pixels, expandedImage.width, expandedImage.height);

    cudaMemcpy(expandedGpuImage, expandedImage.pixels, (expandedImage.width * expandedImage.height), cudaMemcpyHostToDevice);
    cudaCheckError();

    //set global image width for index calculation
    //imageWidth = originalImage.width;

    //==========================================================
    // 3. Step: allocate memory for the results
    //==========================================================
    imgData separateStepCpuImgageWithIndexing(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData separateStepCpuImgageWithInlineIndexing(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData separateStepCpuImgageWithoutIndexing(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData separateStepCpuImgageWithOptimzedCalculation(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData combinedStepsCpuImage(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData combinedStepsCpuImageOptimized(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData combinedStepsCpuImagePow(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);

    imgData separateStepOmpImgageWithIndexing(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData separateStepOmpImgageWithInlineIndexing(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData separateStepOmpImgageWithoutIndexing(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData separateStepOmpImgageWithOptimzedCalculation(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData combinedStepsOmpImage(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData combinedStepsOmpImageOptimized(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);
    imgData combinedStepsOmpImagePow(new byte[originalImage.width * originalImage.height], originalImage.width, originalImage.height);

    byte *gpuImageBasic;
    cudaMallocManaged(&gpuImageBasic, (originalImage.width * originalImage.height) * sizeof(byte));
    cudaCheckError();

    byte* gpuImageOptimizedCalculation;
    cudaMallocManaged(&gpuImageOptimizedCalculation, (originalImage.width * originalImage.height) * sizeof(byte));
    cudaCheckError();

    byte* gpuImageSquare32;
    cudaMallocManaged(&gpuImageSquare32, (originalImage.width * originalImage.height) * sizeof(byte));
    cudaCheckError();

    byte* gpuImageSquare64;
    cudaMallocManaged(&gpuImageSquare64, (originalImage.width * originalImage.height) * sizeof(byte));
    cudaCheckError();

    byte* gpuImageSquare128;
    cudaMallocManaged(&gpuImageSquare128, (originalImage.width * originalImage.height) * sizeof(byte));
    cudaCheckError();

    //==========================================================
    // 4. Step: Executing and measuring for Single Core
    //==========================================================

    //Single core sobel function with indexing and seperated steps (3 total steps) 
    start = std::chrono::system_clock::now();
    separateStepSobelCpuWithIndexing(expandedImage.pixels, separateStepCpuImgageWithIndexing.pixels, expandedImage.width, expandedImage.height, 1);
    times.cpuWithIndexing = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "cpu_1", &separateStepCpuImgageWithIndexing);

    //Single core sobel function with inline indexing and seperated steps (3 total steps) 
    start = std::chrono::system_clock::now();
    separateStepSobelCpuWithInlineIndexing(expandedImage.pixels, separateStepCpuImgageWithInlineIndexing.pixels, expandedImage.width, expandedImage.height, 1);
    times.cpuWithInlineIndexing = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "cpu_2", &separateStepCpuImgageWithInlineIndexing);

    //Single core soble function without indexing and seperated steps (3 total steps) 
    start = std::chrono::system_clock::now();
    separateStepSobelCpuWithoutIndexing(expandedImage.pixels, separateStepCpuImgageWithoutIndexing.pixels, expandedImage.width, expandedImage.height, 1);
    times.cpuWithoutIndexing = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "cpu_3", &separateStepCpuImgageWithoutIndexing);

    //Single core soble function without indexing, seperated steps (3 total steps) and optimized calculations 
    start = std::chrono::system_clock::now();
    separateStepSobelCpuOptimizedCalculation(expandedImage.pixels, separateStepCpuImgageWithOptimzedCalculation.pixels, expandedImage.width, expandedImage.height, 1);
    times.cpuOptimized = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "cpu_4", &separateStepCpuImgageWithOptimzedCalculation);

    //Single core soble function with single loop
    start = std::chrono::system_clock::now();
    combinedStepsSobelCpu(expandedImage.pixels, combinedStepsCpuImage.pixels, expandedImage.width, expandedImage.height, 1);
    times.cpuCombinedSteps = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "cpu_5", &combinedStepsCpuImage);

    //Single core soble function with single loop and optimized calculations
    start = std::chrono::system_clock::now();
    combinedStepsSobelCpuOptimized(expandedImage.pixels, combinedStepsCpuImageOptimized.pixels, expandedImage.width, expandedImage.height, 1);
    times.cpuCombinedStepsOptimized = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "cpu_6", &combinedStepsCpuImageOptimized);

    //Single core soble function with single loop and pow function
    start = std::chrono::system_clock::now();
    combinedStepsSobelCpuPow(expandedImage.pixels, combinedStepsCpuImagePow.pixels, expandedImage.width, expandedImage.height, 1);
    times.cpuPow = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "cpu_7", &combinedStepsCpuImagePow);

    //==========================================================
    // 5. Step: Executing and measuring for multi core
    //==========================================================

    //Multi core sobel function with indexing and seperated steps (3 total steps) 
    start = std::chrono::system_clock::now();
    separateStepSobelCpuWithIndexing(expandedImage.pixels, separateStepOmpImgageWithIndexing.pixels, expandedImage.width, expandedImage.height, maxAvailableCores);
    times.ompWithIndexing = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "omp_1", &separateStepOmpImgageWithIndexing);

    //Multi core sobel function with inline indexing and seperated steps (3 total steps) 
    start = std::chrono::system_clock::now();
    separateStepSobelCpuWithInlineIndexing(expandedImage.pixels, separateStepOmpImgageWithInlineIndexing.pixels, expandedImage.width, expandedImage.height, maxAvailableCores);
    times.ompWithInlineIndexing = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "omp_2", &separateStepOmpImgageWithInlineIndexing);

    //Multi core soble function without indexing and seperated steps (3 total steps) 
    start = std::chrono::system_clock::now();
    separateStepSobelCpuWithoutIndexing(expandedImage.pixels, separateStepOmpImgageWithoutIndexing.pixels, expandedImage.width, expandedImage.height, maxAvailableCores);
    times.ompWithoutIndexing = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "omp_3", &separateStepOmpImgageWithoutIndexing);

    //Multi core soble function without indexing, seperated steps (3 total steps) and optimized calculations 
    start = std::chrono::system_clock::now();
    separateStepSobelCpuOptimizedCalculation(expandedImage.pixels, separateStepOmpImgageWithOptimzedCalculation.pixels, expandedImage.width, expandedImage.height, maxAvailableCores);
    times.ompOptimized = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "omp_4", &separateStepOmpImgageWithOptimzedCalculation);

    //Multi core soble function with single loop
    start = std::chrono::system_clock::now();
    combinedStepsSobelCpu(expandedImage.pixels, combinedStepsOmpImage.pixels, expandedImage.width, expandedImage.height, maxAvailableCores);
    times.ompCombinedSteps = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "omp_5", &combinedStepsOmpImage);

    //Multi core soble function with single loop and optimized calculations
    start = std::chrono::system_clock::now();
    combinedStepsSobelCpuOptimized(expandedImage.pixels, combinedStepsOmpImageOptimized.pixels, expandedImage.width, expandedImage.height, maxAvailableCores);
    times.ompCombinedStepsOptimized = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "omp_6", &combinedStepsOmpImageOptimized);

    //Multi core soble function with single loop and pow function
    start = std::chrono::system_clock::now();
    combinedStepsSobelCpuPow(expandedImage.pixels, combinedStepsOmpImagePow.pixels, expandedImage.width, expandedImage.height, maxAvailableCores);
    times.ompPow = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
    writeImageAndFreeData(image, "omp_7", &combinedStepsOmpImagePow);

    //==========================================================
    // 6. Step: Executing and measuring for GPU
    //==========================================================

    /** set up the dim3's for the gpu to use as arguments (threads per block & num of blocks)**/
    dim3 threadsPerBlock(GRIDSIZE32, GRIDSIZE32, 1);
    int x = ceil(expandedImage.width / (GRIDSIZE32 - 2.0));
    int y = ceil(expandedImage.height / (GRIDSIZE32 - 2.0));

    dim3 numbOfBlocks(x, y);

    /** Run the sobel filter using the CPU **/
    start = std::chrono::system_clock::now();
    sobel_gpu << <numbOfBlocks, threadsPerBlock >> > (expandedGpuImage, gpuImageBasic, expandedImage.width, expandedImage.height);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    times.gpu = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();

    start = std::chrono::system_clock::now();
    sobelGpuOptimized << <numbOfBlocks, threadsPerBlock >> > (expandedGpuImage, gpuImageOptimizedCalculation, expandedImage.width, expandedImage.height);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    times.gpuOptimized = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();

    long size = GRIDSIZE32 * GRIDSIZE32 * sizeof(byte);

    start = std::chrono::system_clock::now();
    sobelGpuSquare32 << <numbOfBlocks, threadsPerBlock, size >> > (expandedGpuImage, gpuImageSquare32, expandedImage.width, expandedImage.height);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    times.gpuSquare32 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();

    size = 64 * 64 * sizeof(byte);
    numbOfBlocks.x = ceil(expandedImage.width / (64 - 2.0));
    numbOfBlocks.y = ceil(expandedImage.height / (64 - 2.0));

    start = std::chrono::system_clock::now();
    sobelGpuSquare64 << <numbOfBlocks, threadsPerBlock, size >> > (expandedGpuImage, gpuImageSquare64, expandedImage.width, expandedImage.height);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    times.gpuSquare64 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();

    size = 128 * 128 * sizeof(byte);
    numbOfBlocks.x = ceil(expandedImage.width / (128 - 2.0));
    numbOfBlocks.y = ceil(expandedImage.height / (128 - 2.0));

    start = std::chrono::system_clock::now();
    sobelGpuSquare128 << <numbOfBlocks, threadsPerBlock, size >> > (expandedGpuImage, gpuImageSquare128, expandedImage.width, expandedImage.height);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    times.gpuSquare128 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();

    printf("CPU with indexing            = %*.2f msec\n", 5, times.cpuWithIndexing / 1000);
    printf("OMP with indexing            = %*.2f msec\n\n", 5, times.ompWithIndexing / 1000);

    printf("CPU with inline indexing     = %*.2f msec\n", 5, times.cpuWithInlineIndexing / 1000);
    printf("OMP with inline indexing     = %*.2f msec\n\n", 5, times.ompWithInlineIndexing / 1000);

    printf("CPU without indexing         = %*.2f msec\n", 5, times.cpuWithoutIndexing / 1000);
    printf("OMP without indexing         = %*.2f msec\n\n", 5, times.ompWithoutIndexing / 1000);

    printf("CPU optimized calc           = %*.2f msec\n", 5, times.cpuOptimized / 1000);
    printf("OMP optimized calc           = %*.2f msec\n\n", 5, times.ompOptimized / 1000);

    printf("CPU combined steps           = %*.2f msec\n", 5, times.cpuCombinedSteps / 1000);
    printf("OMP combined steps           = %*.2f msec\n\n", 5, times.ompCombinedSteps / 1000);

    printf("CPU combined steps optimized = %*.2f msec\n", 5, times.cpuCombinedStepsOptimized / 1000);
    printf("OMP combined steps optimized = %*.2f msec\n\n", 5, times.ompCombinedStepsOptimized / 1000);

    printf("CPU combined steps pow       = %*.2f msec\n", 5, times.cpuPow / 1000);
    printf("OMP combined steps pow       = %*.2f msec\n\n", 5, times.ompPow / 1000);

    printf("GPU basic                    = %*.2f msec\n", 5, times.gpu / 1000);
    printf("GPU calculation optimized    = %*.2f msec\n", 5, times.gpuOptimized / 1000);
    printf("GPU square 32                = %*.2f msec\n", 5, times.gpuSquare32 / 1000);
    printf("GPU square 64                = %*.2f msec\n", 5, times.gpuSquare64 / 1000);
    printf("GPU square 64                = %*.2f msec\n", 5, times.gpuSquare128 / 1000);

    //==========================================================
    // 7. Step: Save GPU result images
    //==========================================================

    /*imgData gpuBasicImage(gpuImageBasic, originalImage.width, originalImage.height);
    imgData gpuOptimizedCalculationImage(gpuImageOptimizedCalculation, originalImage.width, originalImage.height);
    imgData gpuSquare32Image(gpuImageSquare32, originalImage.width, originalImage.height);
    imgData gpuSquare64Image(gpuImageSquare64, originalImage.width, originalImage.height);
    imgData gpuSquare128Image(gpuImageSquare128, originalImage.width, originalImage.height);

    writeImage(image, "gpu_1", gpuBasicImage);
    writeImage(image, "gpu_2", gpuOptimizedCalculationImage);
    writeImage(image, "gpu_3", gpuSquare32Image);
    writeImage(image, "gpu_4", gpuSquare64Image);
    writeImage(image, "gpu_5", gpuSquare128Image);*/

    //==========================================================
    // 8. Step: Free resources
    //==========================================================

    cudaFree(expandedGpuImage); 
    cudaFree(gpuImageBasic); 
    cudaFree(gpuImageOptimizedCalculation);
    cudaFree(gpuImageSquare32);
    cudaFree(gpuImageSquare64);
    cudaFree(gpuImageSquare128);

    delete[] expandedImage.pixels;
    delete[] originalImage.pixels;
    return times;
}

/**
* First implementation of the Sobel operator. In this case, dx and dy are each calculated in a separate run. 
* At the end, the result of dx and dy is also calculated. This results in 3 separate runs. 
* For the access to the one dimensional array an extra getIndex function is used, which allows to access the one dimensional array 
* like a two dimensional array. This implementation does not contain any optimizations and will serve as a basis for comparison 
* to all other Sobel operator implementations.
*/
void separateStepSobelCpuWithIndexing(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores) {
    int* dx = new int[(width - 2) * (height - 2)];
    int* dy = new int[(width - 2) * (height - 2)];

    omp_set_num_threads(maxCores);
    imageWidth = width;

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            dx[(width - 2) * (y - 1) + (x - 1)] = (1 * image[getIndex(x - 1, y - 1)]) + (-1 * image[getIndex(x + 1, y - 1)]) +
                (2 * image[getIndex(x - 1, y)]) + (-2 * image[getIndex(x + 1, y)]) +
                (1 * image[getIndex(x - 1, y + 1)]) + (-1 * image[getIndex(x + 1, y + 1)]);
        }
    }

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            dy[(width - 2) * (y - 1) + (x - 1)] = (1 * image[getIndex(x - 1, y - 1)]) + (2 * image[getIndex(x, y - 1)]) + (1 * image[getIndex(x + 1, y - 1)]) +
                (-1 * image[getIndex(x - 1, y + 1)]) + (-2 * image[getIndex(x, y + 1)]) + (-1 * image[getIndex(x + 1, y + 1)]);
        }
    }

    imageWidth = width - 2;
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            result[getIndex(x - 1, y - 1)] = sqrt((dx[getIndex(x - 1, y - 1)] * dx[getIndex(x - 1, y - 1)]) + (dy[getIndex(x - 1, y - 1)] * dy[getIndex(x - 1, y - 1)]));
        }
    }

    delete[] dx;
    delete[] dy;
}

/**
 * Same implmentation as before, the only optimization is that the indexing function is now a inline function 
 */
void separateStepSobelCpuWithInlineIndexing(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores) {
    int* dx = new int[(width - 2) * (height - 2)];
    int* dy = new int[(width - 2) * (height - 2)];

    omp_set_num_threads(maxCores);
    imageWidth = width;

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            dx[(width - 2) * (y - 1) + (x - 1)] = (1 * image[getIndexInline(x - 1, y - 1)]) + (-1 * image[getIndexInline(x + 1, y - 1)]) +
                (2 * image[getIndexInline(x - 1, y)]) + (-2 * image[getIndexInline(x + 1, y)]) +
                (1 * image[getIndexInline(x - 1, y + 1)]) + (-1 * image[getIndexInline(x + 1, y + 1)]);
        }
    }

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            dy[(width - 2) * (y - 1) + (x - 1)] = (1 * image[getIndexInline(x - 1, y - 1)]) + (2 * image[getIndexInline(x, y - 1)]) + (1 * image[getIndexInline(x + 1, y - 1)]) +
                (-1 * image[getIndexInline(x - 1, y + 1)]) + (-2 * image[getIndexInline(x, y + 1)]) + (-1 * image[getIndexInline(x + 1, y + 1)]);
        }
    }

    imageWidth = width - 2;
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            result[getIndexInline(x - 1, y - 1)] = sqrt((dx[getIndexInline(x - 1, y - 1)] * dx[getIndexInline(x - 1, y - 1)]) + (dy[getIndexInline(x - 1, y - 1)] * dy[getIndexInline(x - 1, y - 1)]));
        }
    }

    delete[] dx;
    delete[] dy;
}

//Same implementation but the calculation is now moved away from an extra function directly into the code
void separateStepSobelCpuWithoutIndexing(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores) {
    int* dx = new int[(width - 2) * (height - 2)];
    int* dy = new int[(width - 2) * (height - 2)];

    omp_set_num_threads(maxCores);

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            dx[(width - 2) * (y - 1) + (x - 1)] = (1 * image[width * (y - 1) + (x - 1)]) + (-1 * image[width * (y - 1) + (x + 1)]) +
                (2 * image[width * y + (x - 1)]) + (-2 * image[width * y + (x + 1)]) +
                (1 * image[width * (y + 1) + (x - 1)]) + (-1 * image[width * (y + 1) + (x + 1)]);
        }
    }

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            dy[(width - 2) * (y - 1) + (x - 1)] = (image[(y - 1) * width + (x - 1)]) + (2 * image[(y - 1) * width + x]) + (image[(y - 1) * width + (x + 1)]) +
                (-1 * image[(y + 1) * width + (x - 1)]) + (-2 * image[(y + 1) * width + x]) + (-1 * image[(y + 1) * width + (x + 1)]);
        }
    }
    
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            result[(y - 1) * (width - 2) + (x - 1)] = sqrt((dx[(y - 1) * (width - 2) + (x - 1)] * dx[(y - 1) * (width - 2) + (x - 1)]) + (dy[(y - 1) * (width - 2) + (x - 1)] * dy[(y - 1) * (width - 2) + (x - 1)]));
        }
    }

    delete[] dx;
    delete[] dy;
}

//optimized calculation
void separateStepSobelCpuOptimizedCalculation(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores) {
    int* dx = new int[(width - 2) * (height - 2)];
    int* dy = new int[(width - 2) * (height - 2)];

    omp_set_num_threads(maxCores);

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        int tmpWidth = width * y;
        int tmp_m = tmpWidth - width;
        int tmp_p = tmpWidth + width;
        for (int x = 1; x < width - 1; x++) {
            dx[(width - 2) * (y - 1) + (x - 1)] = (image[tmp_m + (x - 1)]) + (-image[tmp_m + (x + 1)]) +
                (image[tmpWidth + (x - 1)] << 1) + (-(image[tmpWidth + (x + 1)] << 1)) +
                (image[tmp_p + (x - 1)]) + (-image[tmp_p + (x + 1)]);
        }
    }

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        int tmpWidth = width * y;
        int tmp_m = tmpWidth - width;
        int tmp_p = tmpWidth + width;
        for (int x = 1; x < width - 1; x++) {
            dy[(width - 2) * (y - 1) + (x - 1)] = (image[tmp_m + (x - 1)]) + (image[tmp_m + x] << 1) + (image[tmp_m + (x + 1)]) +
                (-image[tmp_p + (x - 1)]) + (-(image[tmp_p + x] << 1)) + (-image[tmp_p + (x + 1)]);
        }
    }

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        int tmpWidth = (width - 2) * (y - 1);
        for (int x = 1; x < width - 1; x++) {
            result[tmpWidth + (x - 1)] = sqrt((dx[tmpWidth + (x - 1)] * dx[tmpWidth + (x - 1)]) + (dy[tmpWidth + (x - 1)] * dy[tmpWidth + (x - 1)]));
        }
    }

    delete[] dx;
    delete[] dy;
}

void combinedStepsSobelCpu(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores) {
    omp_set_num_threads(maxCores);

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int dx = (image[width * (y - 1) + (x - 1)]) + (-image[width * (y - 1) + (x + 1)]) +
                (2 * image[width * y + (x - 1)]) + (-2 * image[width * y + (x + 1)]) +
                (image[width * (y + 1) + (x - 1)]) + (-image[width * (y + 1) + (x + 1)]);

            int dy = (image[(y - 1) * width + (x - 1)]) + (2 * image[(y - 1) * width + x]) + (image[(y - 1) * width + (x + 1)]) +
                (-image[(y + 1) * width + (x - 1)]) + (-2 * image[(y + 1) * width + x]) + (-image[(y + 1) * width + (x + 1)]);

            result[(width - 2) * (y - 1) + (x - 1)] = sqrt((dx * dx) + (dy * dy));
        }
    }
}

void combinedStepsSobelCpuOptimized(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores) {
    omp_set_num_threads(maxCores);

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        int tmp = width * y;
        int tmp_m = tmp - width;
        int tmp_p = tmp + width;
        for (int x = 1; x < width - 1; x++) {
            int dx = (image[tmp_m + (x - 1)]) + (-image[tmp_m + (x + 1)]) +
                (2 * image[tmp + (x - 1)]) + (-2 * image[tmp + (x + 1)]) +
                (image[tmp_p + (x - 1)]) + (-image[tmp_p + (x + 1)]);

            int dy = (image[tmp_m + (x - 1)]) + (2 * image[tmp_m + x]) + (image[tmp_m + (x + 1)]) +
                (-image[tmp_p + (x - 1)]) + (-2 * image[tmp_p + x]) + (-image[tmp_p + (x + 1)]);

            result[(width - 2) * (y - 1) + (x - 1)] = sqrt((dx * dx) + (dy * dy));
        }
    }
}

void combinedStepsSobelCpuPow(const byte* image, byte* result, const unsigned int width, const unsigned int height, int maxCores) {
    omp_set_num_threads(maxCores);

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        int tmp = width * y;
        int tmp_m = tmp - width;
        int tmp_p = tmp + width;
        for (int x = 1; x < width - 1; x++) {
            int dx = (image[tmp_m + (x - 1)]) + (-image[tmp_m + (x + 1)]) +
                (2 * image[tmp + (x - 1)]) + (-2 * image[tmp + (x + 1)]) +
                (image[tmp_p + (x - 1)]) + (-image[tmp_p + (x + 1)]);

            int dy = (image[tmp_m + (x - 1)]) + (2 * image[tmp_m + x]) + (image[tmp_m + (x + 1)]) +
                (-image[tmp_p + (x - 1)]) + (-2 * image[tmp_p + x]) + (-image[tmp_p + (x + 1)]);

            result[(width - 2) * (y - 1) + (x - 1)] = sqrt(pow(dx, 2) + pow(dy, 2));
        }
    }
}
