#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_THREADS 1024

static int isPowerOfTwo(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

/*
 * OpenMP target-offload analogue of Reduction_OneBlock.cu.
 * CUDA one block <-> OpenMP one team.
 * The explicit tree reduction and barriers are kept for teaching.
 */
int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s n\n", argv[0]);
        printf("Example: %s 256\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (!isPowerOfTwo(n) || n > MAX_THREADS) {
        printf("For this simple one-team example, n must be a power of 2 and n <= %d.\n",
               MAX_THREADS);
        return 1;
    }

    float *A = (float *)malloc((size_t)n * sizeof(float));
    float *result = (float *)malloc(sizeof(float));
    if (!A || !result) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < n; i++)
        A[i] = 1.0f;
    result[0] = 0.0f;

    int ran_on_device = 0;
    double t0, t1;

    #pragma omp target data map(to: A[0:n]) map(from: result[0:1])
    {
        #pragma omp target map(from: ran_on_device)
        ran_on_device = !omp_is_initial_device();

        t0 = omp_get_wtime();

        #pragma omp target teams num_teams(1) thread_limit(n)
        {
            float sdata[MAX_THREADS];

            #pragma omp parallel num_threads(n) shared(sdata)
            {
                int tid = omp_get_thread_num();
                int actual_threads = omp_get_num_threads();
                for (int i = tid; i < n; i += actual_threads)
                    sdata[i] = A[i];

                #pragma omp barrier

                for (int stride = n / 2; stride > 0; stride /= 2) {
                    for (int i = tid; i < stride; i += actual_threads)
                        sdata[i] += sdata[i + stride];

                    #pragma omp barrier
                }

                if (tid == 0)
                    result[0] = sdata[0];
            }
        }

        t1 = omp_get_wtime();
    }

    int ok = fabsf(result[0] - (float)n) < 1.0e-4f;
    printf("Sum = %.1f, expected = %d\n", result[0], n);
    printf("Compute-region time: %.3f ms\n", (t1 - t0) * 1000.0);
    printf("Execution: %s\n", ran_on_device ? "target device" : "host fallback");
    printf("Verification: %s\n", ok ? "PASS" : "FAIL");

    free(A); free(result);
    return ok ? 0 : 1;
}
