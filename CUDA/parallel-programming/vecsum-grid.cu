#include <stdio.h>

#define N	3

__global__ void add( int *a, int *b, int *c ) {
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int tid = threadIdx.x;	// handle the data at this index
	if(tid < N && bidx < N && bidy < N) {
		c[bidx*N*N + bidy*N + tid] = a[bidx*N*N + bidy*N + tid] + b[bidx*N*N + bidy*N + tid];
	}
	printf("blockIDx: %d, blockIDy: %d, threadIDx: %d\n", bidx, bidy, tid);
}

int main( void ) {
	int a[N*N*N], b[N*N*N], c[N*N*N];
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the cpu
	cudaMalloc((void**)&dev_a, N * sizeof(int) * N * N);
	cudaMalloc((void**)&dev_b, N * sizeof(int) * N * N);
	cudaMalloc((void**)&dev_c, N * sizeof(int) * N * N);

	for (int k = 0; k < N; ++k)
	{
		for (int j = 0; j < N; ++j)
		{
			for( int i = 0; i < N; i++ ) {
				a[k*N*N + j*N + i] = -i;
				b[k*N*N + j*N + i] = i * i;
			}
		}
	}

	cudaMemcpy(dev_a, a, N * sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int) * N * N, cudaMemcpyHostToDevice);

	dim3 blockDIM(N, N, 1);

	add<<<blockDIM,N>>>( dev_a, dev_b, dev_c );

	cudaMemcpy(c, dev_c, N * sizeof(int) * N * N, cudaMemcpyDeviceToHost);


	for( int i = 0; i < N*N*N; i++ ){
		printf( "%d + %d = %d\n", a[i], b[i], c[i] );
	}

	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );

	return 0;
}
