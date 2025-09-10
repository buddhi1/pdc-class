#include <stdio.h>
#include <omp.h>

#define ITERS 8

int main() 
{
    int i, j;

    #pragma omp parallel
    {
        // Only one thread creates the tasks
        #pragma omp single
        {
            // First task
            // #pragma omp task
            {
                for (i = 0; i < ITERS; i++) {
                    #pragma omp task
                    printf("Task 1: i = %d (pid=%d)\n", i, omp_get_thread_num());
                }
            }

            // Second task
            // #pragma omp task
            {
                for (j = 0; j < ITERS; j++){
                    #pragma omp task
                    printf("Task 2: j = %d (pid=%d)\n", j, omp_get_thread_num());
                }
            }
        } // implicit taskwait here at the end of single
    }

    return 0;
}
