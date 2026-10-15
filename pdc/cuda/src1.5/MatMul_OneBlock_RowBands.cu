#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* One-dimensional single block.
   Each thread computes one or more COMPLETE ROWS of C.

   If n > number of threads:
       thread 0 computes rows 0, T, 2T, ...
       thread 1 computes rows 1, T+1, 2T+1, ...
       etc.
*/
__global__ void matMulRowBands(float *A, float *B, float *C, int n)
{
    int tid = threadIdx.x;

    for (int row = tid; row < n; row += blockDim.x) {

        for (int col = 0; col < n; col++) {

            float sum = 0.0f;

            for (int k = 0; k < n; k++)
                sum += A[row*n + k] * B[k*n + col];

            C[row*n + col] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 1000;
    int threads = (argc > 2) ? atoi(argv[2]) : 256;

    if (threads > 1024) threads = 1024;

    size_t bytes = (size_t)n*n*sizeof(float);

    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);
    float *hC = (float *)malloc(bytes);

    for (long long i = 0; i < (long long)n*n; i++) {
        hA[i] = 1.0f;
        hB[i] = 2.5f;
    }

    float *A, *B, *C;
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    cudaMemcpy(A, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warm-up */
    matMulRowBands<<<1, threads>>>(A, B, C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matMulRowBands<<<1, threads>>>(A, B, C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC, C, bytes, cudaMemcpyDeviceToHost);

    printf("ONE BLOCK: each thread computes row(s) of C\n");
    printf("Matrix: %d x %d, threads: %d\n", n, n, threads);
    printf("Kernel time: %.3f ms\n", ms);
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

