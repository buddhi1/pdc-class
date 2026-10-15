#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* One CUDA thread computes one C element.
   One block contains the entire matrix, so n*n must be <= 1024. */
__global__ void matMulOneBlock(float *A, float *B, float *C, int n)
{
    int col = threadIdx.x;   // x -> column
    int row = threadIdx.y;   // y -> row

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; k++)
            sum += A[row*n + k] * B[k*n + col];

        C[row*n + col] = sum;
    }
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 32;

    if (n*n > 1024) {
        printf("One-block element version requires n*n <= 1024.\n");
        return 1;
    }

    size_t bytes = (size_t)n*n*sizeof(float);

    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);
    float *hC = (float *)malloc(bytes);

    for (int i = 0; i < n*n; i++) {
        hA[i] = 1.0f;
        hB[i] = 2.5f;
    }

    float *A, *B, *C;
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    cudaMemcpy(A, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(n, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warm-up */
    matMulOneBlock<<<1, block>>>(A, B, C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matMulOneBlock<<<1, block>>>(A, B, C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC, C, bytes, cudaMemcpyDeviceToHost);

    printf("ONE BLOCK: one thread per C element\n");
    printf("Matrix: %d x %d, threads: %d x %d\n", n, n, n, n);
    printf("Kernel time: %.6f ms\n", ms);
    printf("C[0][0] = %.1f, expected = %.1f\n",
           hC[0], 2.5f*n);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}

