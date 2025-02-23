#include <stdio.h>

__global__ void add( int a, int b, int *c ) {
	*c = a + b;
}

int main ( void ) {
	int c;
	int *dev_c;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaMalloc((void**)&dev_c, sizeof(int));
	
	cudaEventRecord(start);
	add<<<1,1>>>( 2, 7, dev_c );
	cudaEventRecord(stop);

	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf( "2 + 7 = %d\n", c );
	printf("\n\nGPU running time: %f\n",milliseconds);

	cudaFree( dev_c );

	return 0;
}
