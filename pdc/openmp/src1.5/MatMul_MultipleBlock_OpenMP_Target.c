#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/*
 * OpenMP target-offload analogue of MatMul_MultipleBlock.cu.
 * CUDA block  <-> OpenMP team
 * CUDA thread <-> OpenMP thread inside a team
 * One thread computes one element of C.
 */
int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s matrix_size block_size\n", argv[0]);
        printf("Example: %s 256 16\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int blockSize = atoi(argv[2]);
    if (n <= 0 || blockSize <= 0) {
        printf("matrix_size and block_size must be positive.\n");
        return 1;
    }

    int threadsPerTeam = blockSize * blockSize;
    if (threadsPerTeam > 1024) {
        printf("block_size^2 must be <= 1024 for this CUDA-like teaching mapping.\n");
        return 1;
    }

    int gridX = (n + blockSize - 1) / blockSize;
    int gridY = (n + blockSize - 1) / blockSize;
    int numTeams = gridX * gridY;

    size_t count = (size_t)n * n;
    float *A = (float *)malloc(count * sizeof(float));
    float *B = (float *)malloc(count * sizeof(float));
    float *C = (float *)malloc(count * sizeof(float));
    if (!A || !B || !C) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    for (size_t i = 0; i < count; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    int ran_on_device = 0;
    double t0, t1;

    #pragma omp target data map(to: A[0:count], B[0:count]) map(from: C[0:count])
    {
        #pragma omp target map(from: ran_on_device)
        ran_on_device = !omp_is_initial_device();

        t0 = omp_get_wtime();

        #pragma omp target teams num_teams(numTeams) thread_limit(threadsPerTeam)
        {
            int team = omp_get_team_num();
            int blockRow = team / gridX;
            int blockCol = team % gridX;

            #pragma omp parallel num_threads(threadsPerTeam)
            {
                int actual_threads = omp_get_num_threads();
                for (int tid = omp_get_thread_num(); tid < threadsPerTeam; tid += actual_threads) {
                    int ty = tid / blockSize;
                    int tx = tid % blockSize;
                    int row = blockRow * blockSize + ty;
                    int col = blockCol * blockSize + tx;

                    if (row < n && col < n) {
                        float sum = 0.0f;
                        for (int k = 0; k < n; k++)
                            sum += A[row * n + k] * B[k * n + col];
                        C[row * n + col] = sum;
                    }
                }
            }
        }

        t1 = omp_get_wtime();
    }

    int errors = 0;
    for (size_t i = 0; i < count; i++) {
        if (fabsf(C[i] - (float)n) > 1.0e-3f) {
            errors++;
            if (errors <= 3)
                printf("Mismatch at C[%zu]: got %f, expected %d\n", i, C[i], n);
        }
    }

    printf("OpenMP multiple-team matrix multiply\n");
    printf("Matrix: %d x %d, CUDA-like block/team: %d x %d\n",
           n, n, blockSize, blockSize);
    printf("Requested team grid: %d x %d = %d teams\n", gridX, gridY, numTeams);
    printf("Requested threads/team: %d\n", threadsPerTeam);
    printf("Compute-region time: %.3f ms\n", (t1 - t0) * 1000.0);
    printf("Execution: %s\n", ran_on_device ? "target device" : "host fallback");
    printf("C[0][0] = %.1f, expected = %d\n", C[0], n);
    printf("Verification: %s\n", errors == 0 ? "PASS" : "FAIL");

    free(A); free(B); free(C);
    return errors == 0 ? 0 : 1;
}
