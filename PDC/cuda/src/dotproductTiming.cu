// Funtionality: Dot product calculation
// compare exection times using shared memory vs not using
// Kernel specification: block size=min(32, (N+255)/256), 256 threads per block, 
#include <stdio.h>

#define imin(a,b) (a<b?a:b)

const int N = 33*1024*20;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

// kernel that uses shared memory
__global__ void dot1(float* a, float* b, float* c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	
	float temp = 0;
	while (tid < N){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	
	// set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	__syncthreads();
	
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

// kernel that does not use shared memory
__global__ void dot_global(float* a, float* b, float* partial) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x; // jump ahead by total #threads
    }

    // store per-thread partial sum in global memory
    partial[index] = temp;
}

__global__ void reduce_global(const float* partial, float* blockResult) {
    int blockStart = blockIdx.x * blockDim.x;   // this block's region in partial[]
    int blockEnd   = min(blockStart + blockDim.x, N);

    float sum = 0.0f;
    // Only one thread per block does the summation
    if (threadIdx.x == 0) {
        for (int i = blockStart; i < blockEnd; i++) {
            sum += partial[i];
        }
        blockResult[blockIdx.x] = sum;  // one value per block
    }
}


int main (void) {
	float *a, *b, c1, c2, *partial_c1, *partial_c2;
	float *dev_a, *dev_b, *dev_partial_c1, *dev_partial_c2;
	cudaEvent_t start1, stop1;

	
	// allocate memory on the cpu side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
	partial_c1 = (float*)malloc(blocksPerGrid*sizeof(float));
	partial_c2 = (float*)malloc(blocksPerGrid*sizeof(float));

	// allocate the memory on the gpu
	cudaMalloc((void**)&dev_a, N*sizeof(float));
	cudaMalloc((void**)&dev_b, N*sizeof(float));
	cudaMalloc((void**)&dev_partial_c1, blocksPerGrid*sizeof(float));
	cudaMalloc((void**)&dev_partial_c2, blocksPerGrid*sizeof(float));
	
	// fill in the host mempory with data
	for(int i=0; i<N; i++) {
		a[i] = i;
		b[i] = i*2;
	}
	
	
	// copy the arrays 'a' and 'b' to the gpu
	cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);
	dot1<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c1);
	cudaEventRecord(stop1);

	cudaEventSynchronize(stop1);
	float kernel1 = 0;
	cudaEventElapsedTime(&kernel1, start1, stop1);
	printf("Kernel1 (shared memory method) execution time: %f ms\n", kernel1);

	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);
	dot_global<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c2);
	reduce_global<<<blocksPerGrid, threadsPerBlock>>>(dev_partial_c2, dev_partial_c2);
	cudaEventRecord(stop1);

	cudaEventSynchronize(stop1);
	kernel1 = 0;
	cudaEventElapsedTime(&kernel1, start1, stop1);
	printf("Kernel2 (global memory method) execution time: %f ms\n", kernel1);

	// copy the array 'c' back from the gpu to the cpu
	cudaMemcpy(partial_c1,dev_partial_c1, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(partial_c2,dev_partial_c2, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

	// finish up on the cpu side
	c1 = 0;
	for(int i=0; i<blocksPerGrid; i++) {
		c1 += partial_c1[i];
	}
	c2 = 0;
	for(int i=0; i<blocksPerGrid; i++) {
		c2 += partial_c2[i];
	}
	
	#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU (global memory method) value %.6g = %.6g?\n", c2, 2*sum_squares((float)(N-1)));
	printf("Does GPU (shared memory method) value %.6g = %.6g?\n", c1, 2*sum_squares((float)(N-1)));

	// free memory on the gpu side
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c1);
	cudaFree(dev_partial_c2);
	// free memory on the cpu side
	free(a);
	free(b);
	free(partial_c1);
	free(partial_c2);
}