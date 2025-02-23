#include <stdio.h>

__global__ void kernel(void){
	printf("From GPU [block id: %d, thread id: %d] Hello, world!\n", blockIdx.x, threadIdx.x);
}

int main( void ) {
	kernel<<<5,10>>>();
	cudaDeviceReset();
	printf("From CPU Hello, world!\n");

	return 0;
}
