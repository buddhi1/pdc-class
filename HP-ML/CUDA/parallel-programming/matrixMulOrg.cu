#include<stdio.h>

#define HEIGHT 2<<10
#define WIDTH 2<<10
// Thread block size
#define BLOCK_SIZE 16


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // timing parameters ready
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

	cudaEventRecord(start);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	cudaEventRecord(stop);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\n\nGPU running time (no Shared memory): %f\n\n",milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}


int main() {
    Matrix A, B, C;
    int i, j;

    A.elements = (float *)malloc(sizeof(float)*HEIGHT*WIDTH);
    B.elements = (float *)malloc(sizeof(float)*HEIGHT*WIDTH);
    C.elements = (float *)malloc(sizeof(float)*HEIGHT*WIDTH);

    // populate matrix A  and B
    for(i=0; i<HEIGHT ; ++i) {
        for(j=0; j<WIDTH; ++j) {
            A.elements[i*HEIGHT + j] = i;
            B.elements[i*HEIGHT + j] = j;
        }
    }
    A.width = WIDTH;
    B.width = WIDTH;
    A.height = HEIGHT;
    B.height = HEIGHT;
    C.width = WIDTH;
    C.height = HEIGHT;

    // call to matrix multiplicaiton host method
    MatMul(A, B, C);

    // print matrices
    // for(i=0; i<HEIGHT ; ++i) {
    //     for(j=0; j<WIDTH; ++j) {
    //         printf("%f ", A.elements[i*HEIGHT + j]);
    //     }
    //     printf("\n");
    // }

    // printf("\n");
    
    // for(i=0; i<HEIGHT ; ++i) {
    //     for(j=0; j<WIDTH; ++j) {
    //         printf("%f ", B.elements[i*HEIGHT + j]);
    //     }
    //     printf("\n");
    // }

    // printf("\n");
    
    // for(i=0; i<HEIGHT ; ++i) {
    //     for(j=0; j<WIDTH; ++j) {
    //         printf("%f ", C.elements[i*HEIGHT + j]);
    //     }
    //     printf("\n");
    // }

    return 0;
}