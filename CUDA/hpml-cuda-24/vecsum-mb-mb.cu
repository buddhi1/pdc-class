// Functionality: Add two given vectors
// Kernel specification: block size=N, N threads per block 
#include <stdio.h>

#define THREADS_PER_BLOCK 3
#define N	THREADS_PER_BLOCK*THREADS_PER_BLOCK

__global__ void add(int *a, int *b, int *c) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;	// handle the data at this index
	int index = bid*blockDim.x+tid;
	if(tid < N)
		c[index] = a[index] + b[index];
	printf("blockIDX: %d threadIDx: %d index: %d\n", bid, tid, index);
}

int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the GPU
	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMalloc((void**)&dev_b, N*sizeof(int));
	cudaMalloc((void**)&dev_c, N*sizeof(int));

	for(int j=0; j<N/THREADS_PER_BLOCK; ++j)
	{
		for(int i=0; i<THREADS_PER_BLOCK; i++){
			a[j*THREADS_PER_BLOCK+i] = -i;
			b[j*THREADS_PER_BLOCK+i] = i*i;
		}
	}

	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);


	for(int i=0; i<N; i++){
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
