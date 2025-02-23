#include <stdio.h>

__global__ void kernel(void){
	printf("From GPU Hello, world!\n");
}

int main( void ) {
	kernel<<<1,10>>>();
	cudaDeviceReset();
	printf("From CPU Hello, world!\n");

	return 0;
}
