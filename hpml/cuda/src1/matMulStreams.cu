/*
Matrix multiplication using single GPU and Multi-GPUs
We use OpenMP to launch parallel GPU threads
Async memory copyin is not must here since CPU threads take care of the parallelism
This method make more sense if there has been heavy CPU heavy load during host to device data off
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono> // Replaces omp.h for high-definition CPU timing

#define WIDTH 1024*1   // Increased size to make Multi-GPU scaling visible
#define BLOCK_SIZE 32

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

// --------------------------------------------------------
// THE SHARED MEMORY KERNEL
// (Reused for both Single and Multi-GPU)
// --------------------------------------------------------
__global__ void matMulGPUShared(float* A, float* B, float* C, int width, int height) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0;

    // Loop over tiles
    for (int tile = 0; tile < width / BLOCK_SIZE; tile++) {
        // Load A: Check if row is valid for this specific chunk height
        if (row < height && (tile * BLOCK_SIZE + threadIdx.x) < width)
            As[threadIdx.y][threadIdx.x] = A[row * width + tile * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B: B is always full width
        if (col < width && (tile * BLOCK_SIZE + threadIdx.y) < width)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * width + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < height && col < width)
        C[row * width + col] = sum;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        printf("Error: This program requires at least 2 GPUs. Found %d.\n", deviceCount);
        return -1;
    }

    printf("Matrix Size: %dx%d\n", WIDTH, WIDTH);
    printf("Hardware: Found %d GPUs.\n", deviceCount);

    // Host Memory
    size_t size = WIDTH * WIDTH * sizeof(float);
    float *h_C_Single = (float*)malloc(size);
    float *h_C_Serial  = (float*)malloc(size);

    float *h_A, *h_B, *h_C_Multi;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C_Multi, size);

    // Initialize
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    printf("\n--- Running Serial CPU ---\n");
    // -------------------------------------------------------
    // 1. SERIAL (C++ High-Res Wall Clock)
    // -------------------------------------------------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    matMulSerial(h_A, h_B, h_C_Serial, WIDTH);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    double serial_time = cpu_duration.count(); 

    // =========================================================================
    // PART 1: SINGLE GPU BASELINE
    // =========================================================================
    printf("--- Running Single GPU (Device 0) ---\n");
    cudaSetDevice(0);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(WIDTH / BLOCK_SIZE, WIDTH / BLOCK_SIZE);

    cudaEvent_t start_single, stop_single;
    cudaEventCreate(&start_single); cudaEventCreate(&stop_single);

    cudaEventRecord(start_single);
    matMulGPUShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, WIDTH, WIDTH);
    cudaEventRecord(stop_single);

    cudaEventSynchronize(stop_single);
    float single_gpu_ms = 0;
    cudaEventElapsedTime(&single_gpu_ms, start_single, stop_single);
    
    cudaMemcpy(h_C_Single, d_C, size, cudaMemcpyDeviceToHost);
    
    // Cleanup Single GPU resources
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start_single); cudaEventDestroy(stop_single);


    // =========================================================================
    // PART 2: MULTI-GPU EXECUTION (Native CUDA Streams)
    // =========================================================================
    printf("--- Running Multi-GPU (%d Devices) ---\n\n", deviceCount);

    // Set up CUDA Events for Multi-GPU Timing on Device 0
    cudaSetDevice(0);
    cudaEvent_t start_multi, stop_multi;
    cudaEventCreate(&start_multi);
    cudaEventCreate(&stop_multi);

    // Arrays to hold streams and device pointers for each GPU
    cudaStream_t* streams = (cudaStream_t*)malloc(deviceCount * sizeof(cudaStream_t));
    float** d_A_local = (float**)malloc(deviceCount * sizeof(float*));
    float** d_B_local = (float**)malloc(deviceCount * sizeof(float*));
    float** d_C_local = (float**)malloc(deviceCount * sizeof(float*));

    // Step 2a: Initialize a stream for each device
    for (int devId = 0; devId < deviceCount; devId++) {
        cudaSetDevice(devId);
        cudaStreamCreate(&streams[devId]);
    }

    // --- START TIMING MULTI-GPU ---
    cudaSetDevice(0);
    cudaEventRecord(start_multi);

    // Step 2b: A single CPU thread rapidly queues work to ALL devices asynchronously
    for (int devId = 0; devId < deviceCount; devId++) {
        cudaSetDevice(devId); 

        int rowsPerGPU = WIDTH / deviceCount;
        int rowOffset  = devId * rowsPerGPU;
        if (devId == deviceCount - 1) {
            rowsPerGPU = WIDTH - rowOffset;
        }

        size_t sizeA_chunk = rowsPerGPU * WIDTH * sizeof(float);
        size_t sizeB_full  = WIDTH * WIDTH * sizeof(float);
        size_t sizeC_chunk = rowsPerGPU * WIDTH * sizeof(float);

        cudaMalloc(&d_A_local[devId], sizeA_chunk);
        cudaMalloc(&d_B_local[devId], sizeB_full);
        cudaMalloc(&d_C_local[devId], sizeC_chunk);

        cudaMemcpyAsync(d_A_local[devId], h_A + (rowOffset * WIDTH), sizeA_chunk, cudaMemcpyHostToDevice, streams[devId]);
        cudaMemcpyAsync(d_B_local[devId], h_B, sizeB_full, cudaMemcpyHostToDevice, streams[devId]);

        dim3 dimGridMulti(WIDTH / BLOCK_SIZE, (rowsPerGPU + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        matMulGPUShared<<<dimGridMulti, dimBlock, 0, streams[devId]>>>(d_A_local[devId], d_B_local[devId], d_C_local[devId], WIDTH, rowsPerGPU);

        cudaMemcpyAsync(h_C_Multi + (rowOffset * WIDTH), d_C_local[devId], sizeC_chunk, cudaMemcpyDeviceToHost, streams[devId]);
    }

    // Step 2c: Now that all work is queued, the CPU waits for each GPU to finish and cleans up
    for (int devId = 0; devId < deviceCount; devId++) {
        cudaSetDevice(devId);
        
        // Block the CPU thread until THIS specific stream is finished
        cudaStreamSynchronize(streams[devId]); 

        cudaFree(d_A_local[devId]);
        cudaFree(d_B_local[devId]);
        cudaFree(d_C_local[devId]);
        cudaStreamDestroy(streams[devId]);
    }

    // --- STOP TIMING MULTI-GPU ---
    // Record the stop event back on Device 0 after all devices have synchronized
    cudaSetDevice(0);
    cudaEventRecord(stop_multi);
    cudaEventSynchronize(stop_multi);

    float multi_gpu_ms = 0;
    cudaEventElapsedTime(&multi_gpu_ms, start_multi, stop_multi);
    cudaEventDestroy(start_multi); cudaEventDestroy(stop_multi);

    free(streams); free(d_A_local); free(d_B_local); free(d_C_local);

    printf("Serial CPU Time: %.2f ms\n", serial_time);
    printf("Single GPU Time: %.2f ms\n", single_gpu_ms);
    printf("Multi-GPU Time:  %.2f ms\n", multi_gpu_ms);

    // =========================================================================
    // VERIFICATION
    // =========================================================================
    int pass_single = matCompare(h_C_Serial, h_C_Single, WIDTH);
    int pass_multi = matCompare(h_C_Serial, h_C_Multi, WIDTH);

    printf("\n--- Correctness Check ---\n");
    printf("Single GPU: %s\n", pass_single ? "PASS" : "FAIL");
    printf("Multi-GPU:  %s\n", pass_multi ? "PASS" : "FAIL");

    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C_Multi);
    free(h_C_Single);  free(h_C_Serial);
    
    return 0;
}