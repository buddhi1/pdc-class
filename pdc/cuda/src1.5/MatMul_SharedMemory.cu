#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

static void checkCuda(cudaError_t err, const char *where)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*
 * Tiled matrix multiplication.
 * Each block computes one tile of C.
 * Each thread loads one A element and one B element into shared memory
 * for each phase, then reuses those shared values for the dot product.
 */
__global__ void matMulShared(const float *A, const float *B, float *C, int n)
{
    int tileSize = blockDim.x;              /* square blocks */
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * tileSize + tx;
    int row = blockIdx.y * tileSize + ty;

    extern __shared__ float shared[];
    float *As = shared;
    float *Bs = shared + tileSize * tileSize;

    float sum = 0.0f;
    int numTiles = (n + tileSize - 1) / tileSize;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * tileSize + tx;
        int bRow = t * tileSize + ty;

        if (row < n && aCol < n)
            As[ty * tileSize + tx] = A[row * n + aCol];
        else
            As[ty * tileSize + tx] = 0.0f;

        if (bRow < n && col < n)
            Bs[ty * tileSize + tx] = B[bRow * n + col];
        else
            Bs[ty * tileSize + tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < tileSize; k++)
            sum += As[ty * tileSize + k] * Bs[k * tileSize + tx];

        /* Finish using this tile before any thread overwrites shared memory. */
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s matrix_size block_size\n", argv[0]);
        printf("Example: %s 256 16\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int blockSize = atoi(argv[2]);

    if (n <= 0 || blockSize <= 0) {
        printf("matrix_size and block_size must be positive.\n");
        return 1;
    }
    if ((long long)blockSize * blockSize > 1024) {
        printf("block_size^2 must be <= 1024 threads per block.\n");
        return 1;
    }

    size_t bytes = (size_t)n * n * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    /* All ones => every output element should equal n. */
    for (size_t i = 0; i < (size_t)n * n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void **)&d_A, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc((void **)&d_B, bytes), "cudaMalloc B");
    checkCuda(cudaMalloc((void **)&d_C, bytes), "cudaMalloc C");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "copy A to device");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "copy B to device");

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((n + blockSize - 1) / blockSize,
                 (n + blockSize - 1) / blockSize);

    size_t sharedBytes = 2ULL * blockSize * blockSize * sizeof(float);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    checkCuda(cudaEventRecord(start), "record start event");
    matMulShared<<<dimGrid, dimBlock, sharedBytes>>>(d_A, d_B, d_C, n);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "wait for kernel");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "copy C to host");

    int errors = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        if (fabsf(h_C[i] - (float)n) > 1.0e-3f) {
            errors++;
            if (errors <= 3)
                printf("Mismatch at C[%zu]: got %f, expected %d\n", i, h_C[i], n);
        }
    }

    printf("Matrix: %d x %d, tile/block: %d x %d, grid: %d x %d\n",
           n, n, blockSize, blockSize, dimGrid.x, dimGrid.y);
    printf("Kernel time: %.3f ms\n", milliseconds);
    printf("C[0][0] = %.1f, expected = %d\n", h_C[0], n);
    printf("Verification: %s\n", errors == 0 ? "PASS" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return errors == 0 ? 0 : 1;
}
