
// Program do vector addition. Use this for cucda-gdb debugging demo
/*
    compile: nvcc -g -G debug.cu -o debug
    debug: cuda-gdb ./debug
    run:./debug 
*/
#include <stdio.h>

#define ARRAY_SIZE 10000
#define THREADS_PER_BLOCK 128

__global__ void addArrays(int *a, int *b, int *c, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // condition omits the extra threads launched
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];
    int *dev_a, *dev_b, *dev_c;

    // Initialize arrays a and b
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc((void**)&dev_a, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, ARRAY_SIZE * sizeof(int));

    // Copy data to device
    cudaMemcpy(dev_a, a, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);


    // Calculate the number of blocks needed
    // uses ceiling for chunk size when using (THREADS_PER_BLOCK - 1)
    dim3 dimGrid((ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    // Launch kernel
    addArrays<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, ARRAY_SIZE);

    // Copy result back to host
    cudaMemcpy(c, dev_c, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
