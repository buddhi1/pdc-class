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

/* One-block tree reduction.  This version intentionally assumes n is a power of 2. */
__global__ void reduceOneBlock(const float *A, float *result, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    sdata[tid] = A[tid];
    __syncthreads();

    for (int stride = n / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        *result = sdata[0];
}

static int isPowerOfTwo(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s n\n", argv[0]);
        printf("Example: %s 256\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (!isPowerOfTwo(n) || n > 1024) {
        printf("For this simple one-block example, n must be a power of 2 and n <= 1024.\n");
        return 1;
    }

    size_t bytes = (size_t)n * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    if (!h_A) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < n; i++)
        h_A[i] = 1.0f;

    float *d_A, *d_result;
    checkCuda(cudaMalloc((void **)&d_A, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc((void **)&d_result, sizeof(float)), "cudaMalloc result");
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "copy A to device");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    checkCuda(cudaEventRecord(start), "record start event");
    reduceOneBlock<<<1, n, bytes>>>(d_A, d_result, n);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "wait for kernel");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");

    float result = 0.0f;
    checkCuda(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "copy result to host");

    printf("Sum = %.1f, expected = %d\n", result, n);
    printf("Kernel time: %.3f ms\n", milliseconds);
    printf("Verification: %s\n", fabsf(result - (float)n) < 1.0e-4f ? "PASS" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_result);
    free(h_A);

    return 0;
}
