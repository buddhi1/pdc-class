// Functionality: Add two given vectors
#include <stdio.h>
#include <cuda_runtime.h>

#define N 27   // total vector size
#define THREADS_PER_BLOCK 3  // number of threads per block

__global__ void add(int *a, int *b, int *c) {
    int blocksPerRow = gridDim.x;
    int blockId = blockIdx.y * blocksPerRow + blockIdx.x;
    int idx = blockId * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
        printf("block(%d,%d) thread(%d) -> idx=%d\n", blockIdx.x, blockIdx.y, threadIdx.x, idx);
    }
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // initialize host arrays
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    // allocate device memory
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // compute grid dimensions
    // int blocksTotal = N / THREADS_PER_BLOCK;
    int blocksTotal = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; //celing of N/THREADS_PER_BLOCK
    int side = (int)sqrt((double)blocksTotal);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 numBlocks(side, side);  // 2D grid

    add<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // print results
	printf("\nResults:\n");
    // print results
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
