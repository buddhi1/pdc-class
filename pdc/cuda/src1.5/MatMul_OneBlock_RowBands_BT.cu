#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* B is supplied to the GPU already TRANSPOSED.

   Original:
       C[row][col] = sum A[row][k] * B[k][col]

   With BT:
       C[row][col] = sum A[row][k] * BT[col][k]

   Therefore, for one thread computing a row of C, BOTH inner-product
   operands are traversed left-to-right in row-major memory.
*/
__global__ void matMulRowBandsBT(float *A, float *BT, float *C, int n)
{
    int tid = threadIdx.x;

    for (int row = tid; row < n; row += blockDim.x) {

        for (int col = 0; col < n; col++) {

            float sum = 0.0f;

            for (int k = 0; k < n; k++)
                sum += A[row*n + k] * BT[col*n + k];

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

    float *hA  = (float *)malloc(bytes);
    float *hB  = (float *)malloc(bytes);
    float *hBT = (float *)malloc(bytes);
    float *hC  = (float *)malloc(bytes);

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            hA[row*n + col] = 1.0f;
            hB[row*n + col] = 2.5f;
        }
    }

    /* Make B transpose on the host.
       BT[col][row] = B[row][col] */
    for (int row = 0; row < n; row++)
        for (int col = 0; col < n; col++)
            hBT[col*n + row] = hB[row*n + col];

    float *A, *BT, *C;
    cudaMalloc(&A, bytes);
    cudaMalloc(&BT, bytes);
    cudaMalloc(&C, bytes);

    cudaMemcpy(A,  hA,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(BT, hBT, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warm-up */
    matMulRowBandsBT<<<1, threads>>>(A, BT, C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matMulRowBandsBT<<<1, threads>>>(A, BT, C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC, C, bytes, cudaMemcpyDeviceToHost);

    printf("ONE BLOCK: row-band version using B transpose\n");
    printf("Matrix: %d x %d, threads: %d\n", n, n, threads);
    printf("MatMul kernel time: %.3f ms\n", ms);
    printf("C[0][0] = %.1f, expected = %.1f\n",
           hC[0], 2.5f*n);

    cudaFree(A);
    cudaFree(BT);
    cudaFree(C);
    free(hA);
    free(hB);
    free(hBT);
    free(hC);

    return 0;
}

