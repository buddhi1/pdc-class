#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/*
 * OpenMP target-offload analogue of MatMul_OneBlock.cu.
 * CUDA: one block, one thread per C element.
 * OpenMP: one team, one requested thread per C element.
 */
int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s matrix_size\n", argv[0]);
        printf("Example: %s 16\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        printf("matrix_size must be positive.\n");
        return 1;
    }

    int threads = n * n;
    if ((long long)threads > 1024) {
        printf("For this one-team teaching example, n*n must be <= 1024.\n");
        printf("Use the multiple-team version for larger matrices.\n");
        return 1;
    }

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

    /* Keep A, B, and C mapped while the compute region runs. */
    #pragma omp target data map(to: A[0:count], B[0:count]) map(from: C[0:count])
    {
        #pragma omp target map(from: ran_on_device)
        ran_on_device = !omp_is_initial_device();

        t0 = omp_get_wtime();

        #pragma omp target teams num_teams(1) thread_limit(threads)
        {
            #pragma omp parallel num_threads(threads)
            {
                int actual_threads = omp_get_num_threads();
                for (int tid = omp_get_thread_num(); tid < threads; tid += actual_threads) {
                    int row = tid / n;
                    int col = tid % n;
                    float sum = 0.0f;
                    for (int k = 0; k < n; k++)
                        sum += A[row * n + k] * B[k * n + col];
                    C[row * n + col] = sum;
                }
            }
        }

        t1 = omp_get_wtime();
    }

    int errors = 0;
    for (size_t i = 0; i < count; i++) {
        if (fabsf(C[i] - (float)n) > 1.0e-4f) {
            errors++;
            if (errors <= 3)
                printf("Mismatch at C[%zu]: got %f, expected %d\n", i, C[i], n);
        }
    }

    printf("OpenMP one-team matrix multiply: %d x %d\n", n, n);
    printf("Requested teams = 1, threads/team = %d\n", threads);
    printf("Compute-region time: %.3f ms\n", (t1 - t0) * 1000.0);
    printf("Execution: %s\n", ran_on_device ? "target device" : "host fallback");
    printf("C[0][0] = %.1f, expected = %d\n", C[0], n);
    printf("Verification: %s\n", errors == 0 ? "PASS" : "FAIL");

    free(A); free(B); free(C);
    return errors == 0 ? 0 : 1;
}
