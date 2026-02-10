/*
compile: nvcc info.cu -o info
run: ./info
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                   \
do {                                                       \
    cudaError_t err = call;                                \
    if (err != cudaSuccess) {                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                \
    }                                                      \
} while (0)

int main(int argc, char *argv[]) {

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("Number of CUDA devices: %d\n\n", deviceCount);
    for (int iDev = 0; iDev < deviceCount; iDev++) {
        cudaDeviceProp iProp;
        CUDA_CHECK(cudaGetDeviceProperties(&iProp, iDev));

        printf("========================================\n");
        printf("Device %d: %s\n", iDev, iProp.name);
        printf("========================================\n");

        printf("Compute capability: %d.%d\n", iProp.major, iProp.minor);
        printf("Number of multiprocessors: %d\n", iProp.multiProcessorCount);
        printf("Total global memory: %4.2f GB\n", iProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Total constant memory: %4.2f KB\n", iProp.totalConstMem / 1024.0);
        printf("Shared memory per block: %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
        printf("Registers per block: %d\n", iProp.regsPerBlock);
        printf("Warp size: %d\n", iProp.warpSize);
        printf("Max threads per block: %d\n", iProp.maxThreadsPerBlock);
        printf("Max threads per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor);
        printf("Max warps per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor / iProp.warpSize);
        printf("\n");
    }

    return 0;
}
