// Functionality: print messages in CPU and GPU. The GPU thread is printing its block id and thread id
// Kernel Specification: block size=5, 10 threads per block
#include <stdio.h>

__global__ void kernel(void){
	printf("From GPU [block id: %d, thread id: %d] Hello, world!\n", blockIdx.x, threadIdx.x);
}

int main(void) {
	kernel<<<3,8>>>();
	cudaDeviceReset();
	printf("From CPU Hello, world!\n");

	return 0;
}
