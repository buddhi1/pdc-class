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

/* One thread computes one element of C.  One block holds the whole matrix. */
__global__ void matMulOneBlock(const float *A, const float *B, float *C, int n)
{
    int col = threadIdx.x; // Note: CUDA maps x along column, and y along rows.
    int row = threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col]; 
		/* C: A[row][k] natural 2-D notation; 
		   CUDA: its row-major flat-memory equivalent: A[row * width + k] 
		   - what cudaMalloc(float*) usually requires
		  */
        C[row * n + col] = sum; //C[row,col]
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s matrix_size\n", argv[0]);
        printf("Example: %s 16\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        printf("matrix_size must be positive.\n");
        return 1;
    }

    /* Current CUDA devices allow at most 1024 threads in one block. */
    if ((long long)n * n > 1024) {
        printf("Cannot use one %d x %d block: %d threads > 1024.\n",
               n, n, n * n);
        printf("Use multiple blocks for this matrix size.\n");
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

    /* Simple values make the answer easy to check: every C element should be n. */
    for (int i = 0; i < n * n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void **)&d_A, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc((void **)&d_B, bytes), "cudaMalloc B");
    checkCuda(cudaMalloc((void **)&d_C, bytes), "cudaMalloc C");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "copy A to device");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "copy B to device");

    dim3 dimBlock(n, n);
    dim3 dimGrid(1, 1);

    matMulOneBlock<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel execution");

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "copy C to host");

    int errors = 0;
    for (int i = 0; i < n * n; i++) {
        if (fabsf(h_C[i] - (float)n) > 1.0e-4f) {
            errors++;
            if (errors <= 3)
                printf("Mismatch at C[%d]: got %f, expected %d\n", i, h_C[i], n);
        }
    }

    printf("C[0][0] = %.1f, expected = %d\n", h_C[0], n);
    printf("Verification: %s\n", errors == 0 ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return errors == 0 ? 0 : 1;
}
