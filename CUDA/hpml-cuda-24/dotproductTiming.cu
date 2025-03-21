// Funtionality: Dot product calculation
// compare exection times using shared memory vs not using
// Kernel specification: block size=min(32, (N+255)/256), 256 threads per block, 
#include <stdio.h>

#define imin(a,b) (a<b?a:b)

const int N = 33*1024*20;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

float dot_product(float *vec1, float *vec2, size_t length) {
    float result = 0.0;

    // Iterate through both vectors and compute the dot product
    for (size_t i = 0; i < length; i++) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

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
__global__ void dot2(float* a, float* b, float* c) {
	float cache[threadsPerBlock];
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


int main (void) {
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	cudaEvent_t start1, stop1;

	
	// allocate memory on the cpu side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid*sizeof(float));
	
	// allocate the memory on the gpu
	cudaMalloc((void**)&dev_a, N*sizeof(float));
	cudaMalloc((void**)&dev_b, N*sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));
	
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
	dot1<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
	cudaEventRecord(stop1);

	cudaEventSynchronize(stop1);
	float kernel1 = 0;
	cudaEventElapsedTime(&kernel1, start1, stop1);
	printf("Kernel1 (using shared memory) execution time: %f ms\n", kernel1);

	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);
	dot2<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
	cudaEventRecord(stop1);

	cudaEventSynchronize(stop1);
	kernel1 = 0;
	cudaEventElapsedTime(&kernel1, start1, stop1);
	printf("Kernel2 (not using shared memory) execution time: %f ms\n", kernel1);

	// copy the array 'c' back from the gpu to the cpu
	cudaMemcpy(partial_c,dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

	// finish up on the cpu side
	c = 0;
	for(int i=0; i<blocksPerGrid; i++) {
		c += partial_c[i];
	}
	
	// #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	// printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float)(N-1)));
	// printf("Does GPU value %.6g = %.6g?\n", c, dot_product(a, b, N));

	
	// free memory on the gpu side
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	
	// free memory on the cpu side
	free(a);
	free(b);
	free(partial_c);
}