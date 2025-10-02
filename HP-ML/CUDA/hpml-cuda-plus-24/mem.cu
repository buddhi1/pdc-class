#include <stdio.h>
#include <cuda_runtime.h>

#define N 128  // Total elements (must be a multiple of warp size)
#define BLOCK_SIZE 64
#define SHIFT 18 

__global__ void kernel(void){
	printf("From GPU [block id: %d, thread id: %d] Hello, world!\n", blockIdx.x, threadIdx.x);
}

__global__ void alignedAccess(int *arr1, int *out) {
    int warpId = threadIdx.x / warpSize;  // Warp index within the block
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        out[idx] = arr1[idx];  // Aligned memory access
    }
    printf("alignedAccess() Thread %d in Block %d belongs to Warp %d access %d from %d\n", idx, blockIdx.x, warpId, idx, idx);
}

__global__ void misalignedAccess(int *arr2, int *out) {
    int warpId = threadIdx.x / warpSize;  // Warp index within the block
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N - 1) {  
        out[idx] = arr2[(idx)+N];  // Misaligned access
    }
    printf("misalignedAccess() Thread %d in Block %d belongs to Warp %d access %d from %d\n", idx, blockIdx.x, warpId, idx, (idx)+N);
}

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
    int *h_arr1 = (int *)malloc(N * sizeof(int));
    int *h_arr2 = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_arr1[i] = i;
        h_arr2[i] = i * 2;  // Different data
    }

    int *d_arr1, *d_arr2, *d_out;
    cudaMalloc(&d_arr1, N * sizeof(int));
    cudaMalloc(&d_arr2, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_arr1, h_arr1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, N * sizeof(int), cudaMemcpyHostToDevice);

	// this kernel prints some info. 
    // It setup the cuda environment for the first time allowing us the time the below 2 kenels excluding cuda setup time
    kernel<<<2,2>>>();

    // Measure execution time
    measureExecutionTime(alignedAccess, d_arr1, d_out, "Aligned Access");
    measureExecutionTime(misalignedAccess, d_arr2, d_out, "Misaligned Access");

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_out);
    free(h_arr1);
    free(h_arr2);
    free(h_out);

    return 0;
}
