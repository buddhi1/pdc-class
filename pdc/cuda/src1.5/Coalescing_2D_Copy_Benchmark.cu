#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Coalesced:
// threadIdx.x changes the COLUMN.
// Consecutive threads therefore access consecutive memory locations.
__global__ void copy_coalesced(float *A, float *B, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n)
        B[row * n + col] = A[row * n + col];
}


// Non-coalesced:
// threadIdx.x changes the ROW.
// Consecutive threads now access locations n floats apart.
__global__ void copy_noncoalesced(float *A, float *B, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n)
        B[row * n + col] = A[row * n + col];
}


int main(int argc, char *argv[])
{
    int n = 8192;
    int reps = 20;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) reps = atoi(argv[2]);

    long long elements = (long long)n * n;
    size_t bytes = elements * sizeof(float);

    float *A, *B;
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);

    // Values do not matter for this copy benchmark.
    cudaMemset(A, 0, bytes);

    // x dimension is one full warp.
    dim3 block(32, 8);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------------- COALESCED ----------------
    copy_coalesced<<<grid, block>>>(A, B, n);
    cudaDeviceSynchronize();   // warm-up

    cudaEventRecord(start);
    for (int r = 0; r < reps; r++)
        copy_coalesced<<<grid, block>>>(A, B, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float coal_ms;
    cudaEventElapsedTime(&coal_ms, start, stop);
    coal_ms /= reps;


    // -------------- NON-COALESCED --------------
    copy_noncoalesced<<<grid, block>>>(A, B, n);
    cudaDeviceSynchronize();   // warm-up

    cudaEventRecord(start);
    for (int r = 0; r < reps; r++)
        copy_noncoalesced<<<grid, block>>>(A, B, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float noncoal_ms;
    cudaEventElapsedTime(&noncoal_ms, start, stop);
    noncoal_ms /= reps;


    printf("Matrix: %d x %d\n", n, n);
    printf("Block: 32 x 8\n");
    printf("Coalesced:      %.3f ms\n", coal_ms);
    printf("Non-coalesced:  %.3f ms\n", noncoal_ms);
    printf("Slowdown:       %.2fx\n", noncoal_ms / coal_ms);

    cudaFree(A);
    cudaFree(B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

