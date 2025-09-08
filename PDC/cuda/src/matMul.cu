// Funtionality: Matrix multiplication using OpenMP, CUDA global memory and CUDA shared memory
// BLOCK_SIZE is also the tile size. Its max size depends on the shared memory size of the GPU. Ex. in Quadro 500 max is 64.
// WIDTH is the matrix width and height. For simplicity matrices are WIDTHxWIDTH

/*
    compile: nvcc -allow-unsupported-compiler -o matMul matMul.cu
    run: ./matMul
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define WIDTH 32
#define BLOCK_SIZE 32

// Function to multiply two matrices in serial mode
void matMulSerial(float* A, float* B, float* C, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            C[i * width + j] = 0;
            for (int k = 0; k < width; k++) {
                C[i * width + j] += A[i * width + k] * B[k * width + j];
            }
        }
    }
}

// Function to multiply two matrices using OpenMP parallel
void matMulOpenMP(float* A, float* B, float* C, int width) {
    #pragma omp parallel for
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            C[i * width + j] = 0;
            for (int k = 0; k < width; k++) {
                C[i * width + j] += A[i * width + k] * B[k * width + j];
            }
        }
    }
} 

// Kernel to multiply two matrices in the GPU
__global__ void matMulGPU(float* A, float* B, float* C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0;
    for (int k = 0; k < width; k++) {
        sum += A[row * width + k] * B[k * width + col];
    }

    C[row * width + col] = sum;
} 

// Kernel to multiply two matrices in the GPU using shared memory
__global__ void matMulGPUShared(float* A, float* B, float* C, int width) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0;

    for (int tile = 0; tile < width / BLOCK_SIZE; tile++) {
        // Load the shared memory with tiles from A and B
        As[threadIdx.y][threadIdx.x] = A[row * width + tile * BLOCK_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * width + col];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * width + col] = sum;
}

// Function to compare two matrices for equality
int matCompare(float* A, float* B, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (A[i * width + j] != B[i * width + j]) {
                return 0; // Matrices are not equal
            }
        }
    }
    return 1; // Matrices are equal
}

// Function to print a matrix
void printMatrix(float* matrix, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

// print sm count
void printSMCount() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        int shared_mem_size = 0;
        cudaDeviceGetAttribute(&shared_mem_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        printf("Total shared memory size: %dkb\n",shared_mem_size);
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d has %d SM(s)\n", i, deviceProp.multiProcessorCount);
    }
}

int main() {
    float *h_A, *h_B, *h_C1, *h_C2, *h_C3, *h_C4, *d_A, *d_B, *d_C1, *d_C2;
    int size = WIDTH * WIDTH * sizeof(float);
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C1 = (float *)malloc(size);
    h_C2 = (float *)malloc(size);
    h_C3 = (float *)malloc(size);
    h_C4 = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_A[i] = 2.0f;
        h_B[i] = 3.0f;
        h_C1[i] = 0.0f;
        h_C2[i] = 0.0f;
        h_C3[i] = 0.0f;
        h_C4[i] = 0.0f;
    }
    printSMCount();

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C1, size);
    cudaMalloc((void**)&d_C2, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid(WIDTH / BLOCK_SIZE, WIDTH / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Perform matrix multiplication in serial mode
    clock_t start_time = clock();
    matMulSerial(h_A, h_B, h_C3, WIDTH);
    clock_t end_time = clock();
    double serial_execution_time = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

    // Perform matrix multiplication using OpenMP
    clock_t omp_start_time = clock();
    matMulOpenMP(h_A, h_B, h_C4, WIDTH);
    clock_t omp_end_time = clock();
    double openmp_execution_time = (double)(omp_end_time - omp_start_time) * 1000.0 / CLOCKS_PER_SEC;

    // Perform matrix multiplication using GPU and global memory only
    start_time = clock();
    matMulGPU<<<dimGrid, dimBlock>>>(d_A, d_B, d_C1, WIDTH);
    cudaDeviceSynchronize();
    end_time = clock();
    double gpu_execution_time = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

    // Copy result from device to host
    cudaMemcpy(h_C1, d_C1, size, cudaMemcpyDeviceToHost);

    // Perform matrix multiplication using GPU and shared memory
    start_time = clock();
    matMulGPUShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C2, WIDTH);
    cudaDeviceSynchronize();
    end_time = clock();
    double gpu_shared_execution_time = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

    // Copy result from device to host
    cudaMemcpy(h_C2, d_C2, size, cudaMemcpyDeviceToHost);


    // Check if the OpenMP result is equal to serial
    int matricesAreEqual1 = matCompare(h_C3, h_C4, WIDTH);
    // Check if the GPU result is equal to serial
    int matricesAreEqual2 = matCompare(h_C3, h_C1, WIDTH);
    // Check if the GPU shared memory result is equal to serial
    int matricesAreEqual3 = matCompare(h_C3, h_C2, WIDTH);

    // Print execution times in milliseconds
    printf("Serial Execution Time: %.6f ms\n", serial_execution_time);
    printf("OpenMP Execution Time: %.6f ms\n", openmp_execution_time);
    printf("GPU Execution Time: %.6f ms\n", gpu_execution_time);
    printf("GPU with Shared Memory Execution Time: %.6f ms\n", gpu_shared_execution_time);

    // Print matrices
    // printf("Matrix A:\n");
    // printMatrix(h_A, WIDTH);

    // printf("Matrix B:\n");
    // printMatrix(h_B, WIDTH);

    // printf("Resulting Matrix C (Serial):\n");
    // printMatrix(h_C1, WIDTH);

    // printf("Resulting Matrix C (GPU):\n");
    // printMatrix(h_C1, WIDTH);

    // printf("Resulting Matrix C (GPU with Shared Memory):\n");
    // printMatrix(h_C2, WIDTH);

    // printf("Resulting Matrix C (OpenMP):\n");
    // printMatrix(h_C3, WIDTH);

    if (matricesAreEqual1) {
        printf("OpenMP result PASS.\n");
    } else {
        printf("OpenMP result FAIL.\n");
    }
    if (matricesAreEqual2) {
        printf("GPU result PASS.\n");
    } else {
        printf("GPU result FAIL.\n");
    }
    if (matricesAreEqual3) {
        printf("GPU shared result PASS.\n");
    } else {
        printf("GPU shared result FAIL.\n");
    }


    // Free allocated memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);

    // Free allocated memory on the host
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    free(h_C3);
    free(h_C4);

    return 0;
}
