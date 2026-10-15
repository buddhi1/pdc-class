#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_TILE 32

/*
 * OpenMP target-offload analogue of MatMul_SharedMemory.cu.
 * Each team computes one tile of C.  As and Bs are one copy per team and are
 * shared by threads in that team.  This mirrors CUDA shared-memory tiling.
 * Portable OpenMP does not guarantee that these arrays occupy NVIDIA shared
 * memory; physical placement is compiler/runtime dependent.
 */
int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s matrix_size tile_size\n", argv[0]);
        printf("Example: %s 256 16\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int tileSize = atoi(argv[2]);
    if (n <= 0 || tileSize <= 0 || tileSize > MAX_TILE) {
        printf("matrix_size must be positive and tile_size must be 1..%d.\n", MAX_TILE);
        return 1;
    }

    int threadsPerTeam = tileSize * tileSize;
    if (threadsPerTeam > 1024) {
        printf("tile_size^2 must be <= 1024 for this CUDA-like teaching mapping.\n");
        return 1;
    }

    int gridX = (n + tileSize - 1) / tileSize;
    int gridY = (n + tileSize - 1) / tileSize;
    int numTeams = gridX * gridY;
    int numTiles = (n + tileSize - 1) / tileSize;

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
            float As[MAX_TILE * MAX_TILE];
            float Bs[MAX_TILE * MAX_TILE];
            float sums[MAX_TILE * MAX_TILE];

            int team = omp_get_team_num();
            int blockRow = team / gridX;
            int blockCol = team % gridX;

            #pragma omp parallel num_threads(threadsPerTeam) shared(As, Bs, sums)
            {
                int tid = omp_get_thread_num();
                int actual_threads = omp_get_num_threads();

                for (int idx = tid; idx < threadsPerTeam; idx += actual_threads)
                    sums[idx] = 0.0f;

                for (int t = 0; t < numTiles; t++) {
                    /* Each actual thread loads as many tile entries as needed. */
                    for (int idx = tid; idx < threadsPerTeam; idx += actual_threads) {
                        int ty = idx / tileSize;
                        int tx = idx % tileSize;
                        int row = blockRow * tileSize + ty;
                        int col = blockCol * tileSize + tx;
                        int aCol = t * tileSize + tx;
                        int bRow = t * tileSize + ty;

                        As[idx] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
                        Bs[idx] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
                    }

                    #pragma omp barrier

                    for (int idx = tid; idx < threadsPerTeam; idx += actual_threads) {
                        int ty = idx / tileSize;
                        int tx = idx % tileSize;
                        for (int k = 0; k < tileSize; k++)
                            sums[idx] += As[ty * tileSize + k] * Bs[k * tileSize + tx];
                    }

                    #pragma omp barrier
                }

                for (int idx = tid; idx < threadsPerTeam; idx += actual_threads) {
                    int row = blockRow * tileSize + idx / tileSize;
                    int col = blockCol * tileSize + idx % tileSize;
                    if (row < n && col < n)
                        C[row * n + col] = sums[idx];
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

    printf("OpenMP tiled/team-local matrix multiply\n");
    printf("Matrix: %d x %d, tile/team: %d x %d, teams: %d\n",
           n, n, tileSize, tileSize, numTeams);
    printf("Compute-region time: %.3f ms\n", (t1 - t0) * 1000.0);
    printf("Execution: %s\n", ran_on_device ? "target device" : "host fallback");
    printf("C[0][0] = %.1f, expected = %d\n", C[0], n);
    printf("Verification: %s\n", errors == 0 ? "PASS" : "FAIL");

    free(A); free(B); free(C);
    return errors == 0 ? 0 : 1;
}
