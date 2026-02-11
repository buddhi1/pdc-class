#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

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
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_Single = (float*)malloc(size);
    float *h_C_Multi  = (float*)malloc(size);
    float *h_C_Serial  = (float*)malloc(size);

    // Initialize
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    printf("\n--- Running Serial CPU ---\n");
    // -------------------------------------------------------
    // 1. SERIAL (High-Res Wall Clock)
    // -------------------------------------------------------
    double start_cpu, end_cpu;
    start_cpu = omp_get_wtime(); // Returns seconds (High Definition Wall Clock)
    matMulSerial(h_A, h_B, h_C_Serial, WIDTH);
    end_cpu = omp_get_wtime();
    double serial_time = (end_cpu - start_cpu) * 1000.0; // Convert to ms

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch with full height
    matMulGPUShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, WIDTH, WIDTH);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float single_gpu_ms = 0;
    cudaEventElapsedTime(&single_gpu_ms, start, stop);
    
    cudaMemcpy(h_C_Single, d_C, size, cudaMemcpyDeviceToHost);
    

    // Cleanup Single GPU resources
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);


    // =========================================================================
    // PART 2: MULTI-GPU EXECUTION
    // =========================================================================
    printf("--- Running Multi-GPU (%d Devices) ---\n\n", deviceCount);

    double omp_start = omp_get_wtime();

    // We use OpenMP to spawn one CPU thread per GPU
    #pragma omp parallel num_threads(deviceCount)
    {
        int devId = omp_get_thread_num();
        cudaSetDevice(devId); 

        // 1. Calculate Decomposed Workload
        // Divide rows evenly among GPUs
        int rowsPerGPU = WIDTH / deviceCount;
        int rowOffset  = devId * rowsPerGPU;
        
        // Handle remainder for odd matrix sizes
        if (devId == deviceCount - 1) {
            rowsPerGPU = WIDTH - rowOffset;
        }

        // 2. Allocate Sliced Memory
        // A and C are sliced (chunks). B is full (replicated).
        size_t sizeA_chunk = rowsPerGPU * WIDTH * sizeof(float);
        size_t sizeB_full  = WIDTH * WIDTH * sizeof(float);
        size_t sizeC_chunk = rowsPerGPU * WIDTH * sizeof(float);

        float *d_A_local, *d_B_local, *d_C_local;
        cudaMalloc(&d_A_local, sizeA_chunk);
        cudaMalloc(&d_B_local, sizeB_full);
        cudaMalloc(&d_C_local, sizeC_chunk);

        // 3. Async Copy
        // Copy only the required rows of A
        cudaMemcpyAsync(d_A_local, h_A + (rowOffset * WIDTH), sizeA_chunk, cudaMemcpyHostToDevice);
        // Copy all of B
        cudaMemcpyAsync(d_B_local, h_B, sizeB_full, cudaMemcpyHostToDevice);

        // 4. Launch Kernel
        // Grid Y is scaled down to match 'rowsPerGPU'
        dim3 dimGridMulti(WIDTH / BLOCK_SIZE, (rowsPerGPU + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Pass 'rowsPerGPU' as the height so kernel knows bounds
        matMulGPUShared<<<dimGridMulti, dimBlock>>>(d_A_local, d_B_local, d_C_local, WIDTH, rowsPerGPU);

        // 5. Copy Result
        cudaMemcpyAsync(h_C_Multi + (rowOffset * WIDTH), d_C_local, sizeC_chunk, cudaMemcpyDeviceToHost);

        // Ensure this device is done before ending the OpenMP thread
        cudaDeviceSynchronize();

        // Cleanup
        cudaFree(d_A_local);
        cudaFree(d_B_local);
        cudaFree(d_C_local);
    } // End Parallel Region

    double omp_end = omp_get_wtime();
    double multi_gpu_ms = (omp_end - omp_start) * 1000.0;

    printf("Serial CPU Time: %.2f ms\n", serial_time);
    printf("Single GPU Time: %.2f ms\n", single_gpu_ms);
    printf("Multi-GPU Time:  %.2f ms\n", multi_gpu_ms);

    // =========================================================================
    // VERIFICATION
    // =========================================================================
    int pass_single = matCompare(h_C_Serial, h_C_Single, WIDTH);
    int pass_multi = matCompare(h_C_Serial, h_C_Multi, WIDTH);

    printf("\n--- Correctness Check ---\n");
    printf("OpenMP:     %s\n", pass_single ? "PASS" : "FAIL");
    printf("GPU Global: %s\n", pass_multi ? "PASS" : "FAIL");

    free(h_A); free(h_B); free(h_C_Single); free(h_C_Multi); free(h_C_Serial);
    return 0;
}