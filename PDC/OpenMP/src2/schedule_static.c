#include <stdio.h>
#include <omp.h>

#define N 16

int main() {
    int i;

    // Create a parallel region
    #pragma omp parallel private(i)
    {
	    // schedule(dynamic, 2) each thread grab next 2 chunks. New chunks are assigned once a thread completes execution
        // schedule(guided, 2) large chunks to small ones. Min size is 2. similar to DYNAMIC, except that chunk size will 	shrink gradually as more threads requesting new work to do
        // RUNTIME: Allow users to specify schedule type during 	execution by setting environment variable OMP_SCHEDULE
        // AUTO: Let compiler and OpenMP library decide

        // Parallelize the loop with a schedule
        #pragma omp for schedule(static, 4)
        for (i = 0; i < N; i++) {
            printf("Thread %d executes iteration %d\n", omp_get_thread_num(), i);
        }
    }

    return 0;
}

