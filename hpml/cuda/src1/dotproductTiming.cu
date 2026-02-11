// Functionality: Dot product calculation
// Comparison: Shared Memory Reduction vs. Global Memory Atomic Add
#include <stdio.h>

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024 * 20;
const int threadsPerBlock = 256;
// Ensure we don't spawn more blocks than needed
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

// --------------------------------------------------------
// Kernel 1: Using Shared Memory (Optimized Reduction)
// --------------------------------------------------------
__global__ void dot_shared(float* a, float* b, float* c) {
    __shared__ float cache[threadsPerBlock];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    // 1. Grid-Stride Loop: Accumulate partial sum in register
    float temp = 0;
    // each thread may have to compute more value based on the input data size
    while (tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;  // jump to next round data
    }
    
    // 2. Load into Shared Memory
    cache[cacheIndex] = temp;
    
    // 3. Barrier: Wait for all threads to write to shared mem
    __syncthreads();
    
    // 4. Parallel Reduction in Shared Memory
    // follows tree reduction hence uses power of 2
    // Requires threadsPerBlock to be a power of 2
    int i = blockDim.x / 2;
    while (i != 0){
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    
    // 5. Write result per block to global memory
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

// --------------------------------------------------------
// Kernel 2: No Shared Memory (Using Global Atomics)
// --------------------------------------------------------
__global__ void dot_no_shared(float* a, float* b, float* c) {
    // No __shared__ array here. We work entirely with registers and global mem.
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 1. Grid-Stride Loop: Accumulate partial sum in register
    float temp = 0;
    // each thread may have to compute more value based on the input data size
    while (tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;  // jump to next round data
    }

    // 2. Direct update to Global Memory
    // This is slower than shared memory reduction due to memory contention
	// read current value, add thread accumulated values and update. All done in atomic fashion
    atomicAdd(&c[blockIdx.x], temp);
}


int main (void) {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Allocate CPU memory
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));
    
    // Allocate GPU memory
    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));
    
    // Initialize Data
    for(int i=0; i<N; i++) {
        a[i] = 1.0f; // Simplified for easy verification
        b[i] = 1.0f;
    }
    
    // Copy to GPU
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create Timing Events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------------------------------------------------------
    // Run Kernel 1 (Shared Memory)
    // -------------------------------------------------------
    cudaEventRecord(start);
    dot_shared<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate CPU result for checking
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    c = 0;
    for(int i=0; i<blocksPerGrid; i++) c += partial_c[i];
    
    printf("Kernel1 (Shared Memory) Time: %.4f ms | Result: %.2f\n", milliseconds, c);

    // -------------------------------------------------------
    // Run Kernel 2 (No Shared Memory - Atomics)
    // -------------------------------------------------------
    
    // IMPORTANT: Reset partial_c to 0 because atomicAdd accumulates!
    cudaMemset(dev_partial_c, 0, blocksPerGrid * sizeof(float));

    cudaEventRecord(start);
    dot_no_shared<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate CPU result for checking
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    c = 0;
    for(int i=0; i<blocksPerGrid; i++) c += partial_c[i];

    printf("Kernel2 (No Shared Mem) Time: %.4f ms | Result: %.2f\n", milliseconds, c);

    // Verification
    printf("Expected Result: %.2f\n", (float)N);

    // Cleanup
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    free(a);
    free(b);
    free(partial_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}