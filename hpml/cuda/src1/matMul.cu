// Funtionality: Matrix multiplication using OpenMP, CUDA global memory and CUDA shared memory
// BLOCK_SIZE is also the tile size. Its max size depends on the shared memory size of the GPU. Ex. in Quadro 500 max is 64.
// WIDTH is the matrix width and height. For simplicity matrices are WIDTHxWIDTH

/*
    compile: nvcc -Xcompiler -fopenmp -o matMul matMul.cu
    run: OMP_NUM_THREADS=8 ./matMul
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define WIDTH 1024
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
    size_t size = WIDTH * WIDTH * sizeof(float);

    // Host Memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_Serial = (float *)malloc(size);
    float *h_C_OMP = (float *)malloc(size);
    float *h_C_GPU = (float *)malloc(size);
    float *h_C_Shared = (float *)malloc(size);

    // Initialize
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_A[i] = 2.0f;
        h_B[i] = 3.0f;
    }
    printSMCount();

    // Device Memory
    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C1, size);
    cudaMalloc((void**)&d_C2, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(WIDTH / BLOCK_SIZE, WIDTH / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Variables for timing
    double start_cpu, end_cpu;
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_milliseconds = 0;

    // Create CUDA Events
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // -------------------------------------------------------
    // 1. SERIAL (High-Res Wall Clock)
    // -------------------------------------------------------
    printf("Running Serial...\n");
    start_cpu = omp_get_wtime(); // Returns seconds (High Definition Wall Clock)
    matMulSerial(h_A, h_B, h_C_Serial, WIDTH);
    end_cpu = omp_get_wtime();
    double serial_time = (end_cpu - start_cpu) * 1000.0; // Convert to ms

    // -------------------------------------------------------
    // 2. OpenMP (High-Res Wall Clock)
    // -------------------------------------------------------
    printf("Running OpenMP...\n");
    start_cpu = omp_get_wtime();
    matMulOpenMP(h_A, h_B, h_C_OMP, WIDTH);
    end_cpu = omp_get_wtime();
    double openmp_time = (end_cpu - start_cpu) * 1000.0;

    // -------------------------------------------------------
    // 3. GPU Global (CUDA Events)
    // -------------------------------------------------------
    printf("Running GPU Global...\n");
    cudaEventRecord(start_gpu); // Record start
    matMulGPU<<<dimGrid, dimBlock>>>(d_A, d_B, d_C1, WIDTH);
    cudaEventRecord(stop_gpu);  // Record stop

    cudaEventSynchronize(stop_gpu); // Wait for GPU to finish
    cudaEventElapsedTime(&gpu_milliseconds, start_gpu, stop_gpu); // Calculate delta
    double gpu_global_time = (double)gpu_milliseconds;

    cudaMemcpy(h_C_GPU, d_C1, size, cudaMemcpyDeviceToHost);

    // -------------------------------------------------------
    // 4. GPU Shared (CUDA Events)
    // -------------------------------------------------------
    printf("Running GPU Shared...\n");
    cudaEventRecord(start_gpu);
    matMulGPUShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C2, WIDTH);
    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_milliseconds, start_gpu, stop_gpu);
    double gpu_shared_time = (double)gpu_milliseconds;

    cudaMemcpy(h_C_Shared, d_C2, size, cudaMemcpyDeviceToHost);

    // -------------------------------------------------------
    // Validation & Output
    // -------------------------------------------------------
    int pass_omp = matCompare(h_C_Serial, h_C_OMP, WIDTH);
    int pass_gpu = matCompare(h_C_Serial, h_C_GPU, WIDTH);
    int pass_shm = matCompare(h_C_Serial, h_C_Shared, WIDTH);

    printf("\n--- Performance Results ---\n");
    printf("Serial Time:       %8.4f ms\n", serial_time);
    printf("OpenMP Time:       %8.4f ms\n", openmp_time);
    printf("GPU Global Time:   %8.4f ms\n", gpu_global_time);
    printf("GPU Shared Time:   %8.4f ms\n", gpu_shared_time);

    printf("\n--- Correctness Check ---\n");
    printf("OpenMP:     %s\n", pass_omp ? "PASS" : "FAIL");
    printf("GPU Global: %s\n", pass_gpu ? "PASS" : "FAIL");
    printf("GPU Shared: %s\n", pass_shm ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    free(h_A); free(h_B); free(h_C_Serial); free(h_C_OMP); free(h_C_GPU); free(h_C_Shared);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
