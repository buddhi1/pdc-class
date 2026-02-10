/*
This program demonstrates memory coalesced Access vs strided access


compile: nvcc  mem.cu -o  mem
Run: ./mem
Profile: ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld ./mem
*/

/*
Profiling result

==PROF== Connected to process 376615 (/home/buddhi/pdc-class/HP-ML/CUDA/hpml-cuda-plus-24/mem)
Benchmarking Strided Memory Access...
Threads: 4194304, Stride: 32 (Accessing data 128 bytes apart)
----------------------------------------------------------------
==PROF== Profiling "coalescedKernel" - 0: 0%....50%....100% - 3 passes
==PROF== Profiling "coalescedKernel" - 1: 0%....50%....100% - 3 passes
Coalesced (Stride 1)      Time:  206.786 ms  |  Effective BW:     0.16 GB/s
==PROF== Profiling "stridedKernel" - 2: 0%....50%....100% - 3 passes
Strided (Stride 32)       Time:  205.026 ms  |  Effective BW:     0.16 GB/s
==PROF== Disconnected from process 376615
[376615] mem@127.0.0.1
  coalescedKernel(int *, int *, int) (16384, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    ----------------------------------------------------------------------- ----------- ------------
    Metric Name                                                             Metric Unit Metric Value
    ----------------------------------------------------------------------- ----------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.max_rate      sector           32
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.pct                %        12.50
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio         sector            4
    ----------------------------------------------------------------------- ----------- ------------

  coalescedKernel(int *, int *, int) (16384, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    ----------------------------------------------------------------------- ----------- ------------
    Metric Name                                                             Metric Unit Metric Value
    ----------------------------------------------------------------------- ----------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.max_rate      sector           32
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.pct                %        12.50
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio         sector            4
    ----------------------------------------------------------------------- ----------- ------------

  stridedKernel(int *, int *, int) (16384, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    ----------------------------------------------------------------------- ----------- ------------
    Metric Name                                                             Metric Unit Metric Value
    ----------------------------------------------------------------------- ----------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.max_rate      sector           32
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.pct                %          100
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio         sector           32
    ----------------------------------------------------------------------- ----------- ------------

    For the following config
    #define N_THREADS (1 << 22)
    #define BLOCK_SIZE 256
    #define STRIDE 32     
    
    Our warp size=32
    int size=4B
    data used per warp = 32x4B=128B

    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio         sector : 
        in coalescedKernel() - 4 (Warp requested 32Bx4=128B means 1 transaction per )
        in stridedKernel() - 32 (Warp requested)
*/


#include <stdio.h>
#include <cuda_runtime.h>

// Number of elements to process (Threads)
#define N_THREADS (1 << 22) // ~4 Million threads
#define BLOCK_SIZE 256
#define STRIDE 32           // Stride of 32 integers (128 bytes)

// ==========================================
// Kernel 1: Coalesced Access (Fast)
// ==========================================
// Reads: input[0], input[1], input[2]...
// Memory Transactions: 1 per warp
__global__ void coalescedKernel(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx]; 
    }
}

// ==========================================
// Kernel 2: Strided Access (Slow)
// ==========================================
// Reads: input[0], input[32], input[64]...
// Memory Transactions: 32 per warp (Worst Case)
__global__ void stridedKernel(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each thread reads a value 128 bytes apart from its neighbor.
        // This defeats the cache line coalescing mechanism completely.
        output[idx] = input[idx * STRIDE]; 
    }
}

void measureExecutionTime(void (*kernel)(int *, int *, int), int *d_in, int *d_out, int n, const char *kernelName) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(start);
    kernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    // Calculate Effective Bandwidth (GB/s)
    // We read N ints and write N ints. Total bytes transferred is 2 * N * 4.
    // Note: The strided kernel actually fetches MORE data from DRAM (overhead), 
    // but "Effective Bandwidth" measures useful work done.
    double bandwidth = (2.0 * 4.0 * n) / (time * 1e6); 
    
    printf("%-25s Time: %8.3f ms  |  Effective BW: %8.2f GB/s\n", kernelName, time, bandwidth);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // 1. Determine sizes
    // The "input" array for the Strided kernel must be much larger
    // because we skip 32 elements for every 1 element we read.
    size_t data_size_linear = N_THREADS * sizeof(int);
    size_t data_size_strided = N_THREADS * STRIDE * sizeof(int);

    // 2. Allocate Host Memory
    int *h_in = (int *)malloc(data_size_strided); // Large enough for stride
    int *h_out = (int *)malloc(data_size_linear);

    // Initialize with dummy data
    for (int i = 0; i < N_THREADS * STRIDE; i++) {
        h_in[i] = 1;
    }

    // 3. Allocate Device Memory
    int *d_in, *d_out;
    cudaMalloc(&d_in, data_size_strided); // Must accomodate largest access
    cudaMalloc(&d_out, data_size_linear);

    cudaMemcpy(d_in, h_in, data_size_strided, cudaMemcpyHostToDevice);

    printf("Benchmarking Strided Memory Access...\n");
    printf("Threads: %d, Stride: %d (Accessing data 128 bytes apart)\n", N_THREADS, STRIDE);
    printf("----------------------------------------------------------------\n");

    // Warmup
    coalescedKernel<<< (N_THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(d_in, d_out, N_THREADS);
    cudaDeviceSynchronize();

    // Measure
    measureExecutionTime(coalescedKernel, d_in, d_out, N_THREADS, "Coalesced (Stride 1)");
    measureExecutionTime(stridedKernel, d_in, d_out, N_THREADS, "Strided (Stride 32)");

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}