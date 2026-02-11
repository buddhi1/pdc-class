/*
This program demonstrates bank conflict vs no bank conflicts


compile: nvcc  bankconf.cu -o  bankconf
Run: ./bankconf
Profile: ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./bankconf
*/


/*
Profiling result

==PROF== Connected to process 375702 (/home/buddhi/pdc-class/HP-ML/CUDA/hpml-cuda-plus-24/bankconf)
==PROF== Profiling "warmup()" - 0: 0%....50%....100% - 1 pass
Comparing Shared Memory Access Patterns
------------------------------------------
==PROF== Profiling "noBankConflictKernel(int *)" - 1: 0%....50%....100% - 1 pass
No Bank Conflict     Execution Time: 138.635513 ms
==PROF== Profiling "bankConflictKernel(int *)" - 2: 0%....50%....100% - 1 pass
32-Way Conflict      Execution Time: 162.863449 ms
==PROF== Disconnected from process 375702
[375702] bankconf@127.0.0.1
  warmup() (1, 1, 1)x(1, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    -------------------------------------------------------- ----------- ------------

  noBankConflictKernel(int *) (3907, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    -------------------------------------------------------- ----------- ------------

  bankConflictKernel(int *) (3907, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum              968,936,000
    -------------------------------------------------------- ----------- ------------

    For the following config
    #define N 1000000
    #define BLOCK_SIZE 256
    #define ITERATIONS 1000
    #define SHMEM_SIZE 1024

    968,936,000 bank conflicts 
*/


#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000  // Increased N for better measurement stability
#define BLOCK_SIZE 256
#define ITERATIONS 1000 // Increase load to make memory latency visible
#define SHMEM_SIZE 1024

__global__ void warmup(void){
    // Empty kernel to handle CUDA context initialization overhead
}

// ==========================================
// Kernel 1: Contiguous Access (Fast)
// ==========================================
// Threads 0, 1, 2... access Banks 0, 1, 2...
__global__ void noBankConflictKernel(int *out) {
    // 32 banks * 32 stride = SHMEM_SIZE ints ensures enough space for stride tricks
    __shared__ volatile int sharedMem[SHMEM_SIZE]; 

    int tid = threadIdx.x;
    
    // Initialize shared memory (Linear - No conflict)
    // We wrap indices to stay within SHMEM_SIZE size
    sharedMem[tid] = tid; 
    __syncthreads();

    // The index is just the thread ID. 
    // T0->Bank0, T1->Bank1... T31->Bank31.
    // This is perfect parallel access.
    int index = tid; 
    
    int val = 0;
    
    // Heavy loop to highlight memory access time
    for (int i = 0; i < ITERATIONS; i++) {
        // We use a dummy dependency (val) to ensure the compiler 
        // doesn't strip the loop, but we keep the index simple.
        val += sharedMem[index]; 
    }

    if (tid == 0) out[blockIdx.x] = val; // Prevent dead-code elimination
}

// ==========================================
// Kernel 2: Stride-32 Access (Slow)
// ==========================================
// Threads 0, 1, 2... ALL access Bank 0 (at diff addresses)
__global__ void bankConflictKernel(int *out) {
    __shared__ volatile int sharedMem[SHMEM_SIZE];

    int tid = threadIdx.x;

    sharedMem[tid] = tid;
    __syncthreads();

    // CONFLICT LOGIC:
    // We want all threads in a warp to hit the same bank.
    // Bank index = (Address) % 32.
    // If we multiply tid * 32, every result is a multiple of 32.
    // Multiples of 32 always land in Bank 0.
    // We use % SHMEM_SIZE to keep it inside the array bounds.
    // index is 0, 32, 64, etc
    int index = (tid * 32) % SHMEM_SIZE;

    int val = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        val += sharedMem[index];
    }

    if (tid == 0) out[blockIdx.x] = val;
}

void measureExecutionTime(void (*kernel)(int *), int *d_out, const char *kernelName) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_out);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%-20s Execution Time: %f ms\n", kernelName, time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int *d_out;
    cudaMalloc(&d_out, N * sizeof(int));

    // Warmup to remove startup overhead from measurements
    warmup<<<1,1>>>();
    cudaDeviceSynchronize();

    printf("Comparing Shared Memory Access Patterns\n");
    printf("------------------------------------------\n");

    measureExecutionTime(noBankConflictKernel, d_out, "No Bank Conflict");
    measureExecutionTime(bankConflictKernel, d_out, "32-Way Conflict");

    cudaFree(d_out);
    return 0;
}
