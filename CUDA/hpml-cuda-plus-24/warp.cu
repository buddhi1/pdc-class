#include <stdio.h>

__global__ void warpInfoKernel() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadIdx.x / warpSize;  // Warp index within the block

    // Print which warp the thread belongs to
    // printf("Thread %d in Block %d belongs to Warp %d\n", threadId, blockIdx.x, warpId);

    // Introducing warp divergence
    if (threadIdx.x % 2 == 0) {
        printf("Thread %d: Even index, executing branch A Warp %d\n", threadId, warpId);
    } else {
        printf("Thread %d: Odd index, executing branch B Warp %d\n", threadId, warpId);
    }
}

__global__ void warpInfoKerne2() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadIdx.x / warpSize;  // Warp index within the block

    // Print which warp the thread belongs to
    // printf("Thread %d in Block %d belongs to Warp %d\n", threadId, blockIdx.x, warpId);

    // interleave data using warp
    if ((threadIdx.x/warpSize) % 2 == 0) {
        printf("Thread %d: Even index, executing branch A Warp %d\n", threadId, warpId);
    } else {
        printf("Thread %d: Odd index, executing branch B Warp %d\n", threadId, warpId);
    }
}

int main() {
    int numBlocks = 2;
    int threadsPerBlock = 64; // More than one warp per block

    warpInfoKernel<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize(); // Wait for kernel completion
    printf("************** Interleave warp *****************\n");

    warpInfoKerne2<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize(); // Wait for kernel completion

    return 0;
}
