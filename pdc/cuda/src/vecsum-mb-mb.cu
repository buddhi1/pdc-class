// Functionality: Add two given vectors
#include <stdio.h>
#include <cuda_runtime.h>

#define N  16   // total number of elements in the vector
#define THREADS_PER_BLOCK 4  // number of threads per block. use multiples 0f 32 for better performance

__global__ void add(int *a, int *b, int *c, int n) {
	int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;   // flat index across the grid

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        printf("block=%d thread=%d -> idx=%d: %d + %d = %d\n", bid, tid, idx, a[idx], b[idx], c[idx]);
    }
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate device memory
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // initialize host arrays
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    // copy host arrays to device
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // choose block and grid size
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; //celing of N/THREADS_PER_BLOCK

    // launch kernel
    add<<<blocks, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c, N);

    // copy results back
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nResults:\n");
    // print results
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // cleanup
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

