
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

#ifndef OUTS
#define OUTS 2
#endif

#define CHECK_CUDA(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

__global__ void matMulSharedRegisterTiled(const float *__restrict__ A,
                                          const float *__restrict__ B,
                                          float *__restrict__ C,
                                          int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][OUTS * TILE];

    const int tx = threadIdx.x;        // x -> column
    const int ty = threadIdx.y;        // y -> row

    const int row = blockIdx.y * TILE + ty;
    const int baseCol = blockIdx.x * (OUTS * TILE) + tx;

    float sum[OUTS];

    #pragma unroll
    for (int r = 0; r < OUTS; r++)
        sum[r] = 0.0f;

    const int phases = (n + TILE - 1) / TILE;

    for (int t = 0; t < phases; t++) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        // Load one TILE x TILE tile of A.
        As[ty][tx] = (row < n && aCol < n)
                   ? A[(size_t)row * n + aCol]
                   : 0.0f;

        // Load OUTS adjacent TILE-wide tiles of B.
        // For each r, threads with consecutive tx load consecutive columns:
        // therefore each of these global loads is coalesced.
        #pragma unroll
        for (int r = 0; r < OUTS; r++) {
            const int col = baseCol + r * TILE;

            Bs[ty][tx + r * TILE] =
                (bRow < n && col < n)
                ? B[(size_t)bRow * n + col]
                : 0.0f;
        }

        __syncthreads();

        // Each thread computes OUTS C values and keeps all partial sums
        // in registers.
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            const float a = As[ty][k];

            #pragma unroll
            for (int r = 0; r < OUTS; r++)
                sum[r] += a * Bs[k][tx + r * TILE];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < OUTS; r++) {
        const int col = baseCol + r * TILE;
        if (row < n && col < n)
            C[(size_t)row * n + col] = sum[r];
    }
}

static void initialize(float *A, float *B, int n)
{
    const size_t count = (size_t)n * n;
    for (size_t i = 0; i < count; i++) {
        A[i] = 1.0f;
        B[i] = 2.5f;
    }
}

static int verify(const float *C, int n)
{
    const float expected = 2.5f * n;
    int ok = 1;

    int rows[] = {0, n/2, n-1};
    int cols[] = {0, n/3, n-1};

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            int r = rows[a], c = cols[b];
            float got = C[(size_t)r * n + c];
            if (fabsf(got - expected) > 1e-2f * fmaxf(1.0f, expected)) {
                printf("Mismatch at C[%d][%d]: got %.6f, expected %.6f\n",
                       r, c, got, expected);
                ok = 0;
            }
        }
    }

    printf("C[0][0] = %.1f, expected = %.1f\n", C[0], expected);
    printf("Verification: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv)
{
    const int n = (argc > 1) ? atoi(argv[1]) : 2048;
    if (n <= 0) return 1;

    const size_t count = (size_t)n * n;
    const size_t bytes = count * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    if (!hA || !hB || !hC) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    initialize(hA, hB, n);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + OUTS*TILE - 1) / (OUTS*TILE),
              (n + TILE - 1) / TILE);

    // Warm-up.
    matMulSharedRegisterTiled<<<grid, block>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matMulSharedRegisterTiled<<<grid, block>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    printf("Parameterized register-tiled shared-memory matrix multiply\n");
    printf("Matrix: %d x %d\n", n, n);
    printf("TILE=%d, OUTS=%d\n", TILE, OUTS);
    printf("Block: %d x %d threads\n", TILE, TILE);
    printf("Output tile per block: %d rows x %d columns\n",
           TILE, OUTS*TILE);
    printf("Grid: %d x %d\n", grid.x, grid.y);
    printf("Kernel time: %.3f ms\n", ms);

    verify(hC, n);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);

    return 0;
}

