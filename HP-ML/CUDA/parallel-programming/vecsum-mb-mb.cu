#include <stdio.h>

#define N	10

__global__ void add( int *a, int *b, int *c ) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;	// handle the data at this index
	if(tid < N)
		c[bid][tid] = a[bid][tid] + b[bid][tid];
	printf("threadIDx: %d\n", tid);
}

int main( void ) {
	int a[N][N], b[N][N], c[N][N];
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the cpu
	cudaMalloc( (void**)&dev_a, N * sizeof(int));
	cudaMalloc( (void**)&dev_b, N * sizeof(int));
	cudaMalloc( (void**)&dev_c, N * sizeof(int));

	for (int j = 0; j < N; ++j)
	{
		for( int i = 0; i < N; i++ ) {
			a[j][i] = -i;
			b[j][i] = i * i;
		}
	}

	cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

		add<<<N,N>>>( dev_a, dev_b, dev_c );

	cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);


	for( int i = 0; i < N; i++ ){
		printf( "%d + %d = %d\n", a[i], b[i], c[i] );
	}

	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );

	return 0;
}
