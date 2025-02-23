#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000  // Total elements
#define BLOCK_SIZE 256
#define BANKS 8  // Number of shared memory banks

__global__ void kernel(void){
	printf("From GPU [block id: %d, thread id: %d] Hello, world!\n", blockIdx.x, threadIdx.x);
}

// Kernel with **no bank conflicts**
__global__ void noBankConflictKernel(int *arr, int *out) {
    __shared__ int sharedMem[N];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int bankIndex = threadIdx.x; // Accesses different banks

    if (idx < N) {
        sharedMem[bankIndex] = arr[idx];  // No conflicts (separate banks)
        __syncthreads();
        out[idx] = sharedMem[bankIndex];  // Read from separate banks
    }
}

// Kernel with **bank conflicts**
__global__ void bankConflictKernel(int *arr, int *out) {
    __shared__ int sharedMem[N];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int bankIndex = threadIdx.x % BANKS; // Forces multiple threads into the same bank

    if (idx < N) {
        sharedMem[bankIndex] = arr[idx];  // Bank conflicts occur here
        __syncthreads();
        out[idx] = sharedMem[bankIndex];  // Read from the same conflicted bank
    }
}

// Function to measure execution time
void measureExecutionTime(void (*kernel)(int *, int *), int *d_arr, int *d_out, const char *kernelName) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_arr, d_out);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%s Execution Time: %f ms\n", kernelName, time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int *h_arr = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) h_arr[i] = i;

    int *d_arr, *d_out;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // this kernel prints some info. 
    // It setup the cuda environment for the first time allowing us the time the below 2 kenels excluding cuda setup time
    kernel<<<2,2>>>();

    // Measure execution time
    measureExecutionTime(noBankConflictKernel, d_arr, d_out, "No Bank Conflict");
    measureExecutionTime(bankConflictKernel, d_arr, d_out, "Bank Conflict");

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_arr);
    cudaFree(d_out);
    free(h_arr);
    free(h_out);

    return 0;
}
