/*
This program demonstrates warp divergence


compile: nvcc  warp.cu -o  warp
Run: ./warp
Profile: ncu --section WarpStateStats ./warp
*/

/*
profile results

  divergentKernel(float *) (7812, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        30.64
    Warp Cycles Per Executed Instruction           cycle        30.64
    Avg. Active Threads Per Warp                                16.00
    Avg. Not Predicated Off Threads Per Warp                    14.15
    ---------------------------------------- ----------- ------------

  sortedKernel(float *) (7812, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        30.07
    Warp Cycles Per Executed Instruction           cycle        30.07
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    28.30
    ---------------------------------------- ----------- ------------


    For the following config
    #define N 1000000 
    #define BLOCK_SIZE 128
    #define ITERATIONS 100000 

     Avg. Active Threads Per Warp
        divergentKernel(): 16
        sortedKernel(): 32 (perfect number)
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000 
#define BLOCK_SIZE 128
// Increase workload to ensure compute time > memory time
#define ITERATIONS 100000 

__global__ void warmup(void){
    // Empty kernel to handle CUDA context initialization overhead
}

__device__ float heavyMath(float x) {
    // 'volatile' prevents the compiler from optimizing the loop away
    volatile float v = x; 
    for (int i = 0; i < ITERATIONS; i++) {
        // Simple math that changes v, so it can't be skipped
        v = v * v + 0.0001f; 
        if (v > 1000.0f) v = x; // Reset to keep it running
    }
    return v;
}

// 1. Divergent Kernel (Slow)
__global__ void divergentKernel(float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val;
    // Even/Odd divergence
    if (idx % 2 == 0) {
        val = heavyMath(2.0f);
    } else {
        val = heavyMath(3.0f);
    }
    out[idx] = val;
}

// 2. Sorted Kernel (Fast)
__global__ void sortedKernel(float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val;
    int warpId = threadIdx.x / 32;

    // Warp-level branching (No divergence)
    if (warpId % 2 == 0) {
        val = heavyMath(2.0f);
    } else {
        val = heavyMath(3.0f);
    }
    out[idx] = val;
}

int main() {
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // Warmup to remove startup overhead from measurements
    warmup<<<1,1>>>();
    cudaDeviceSynchronize();

    // 1. Run Divergent
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    divergentKernel<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Divergent Time: %f ms\n", time);

    // 2. Run Sorted
    cudaEventRecord(start);
    sortedKernel<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Sorted Time:    %f ms\n", time);

    cudaFree(d_out);
    return 0;
}