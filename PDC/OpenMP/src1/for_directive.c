#include <stdio.h>
#include <omp.h>

int main() 
{
    int i, count=10;
    double start, end;

    // record start time
    start = omp_get_wtime();

    /*
     * create threads, each takes some of the iterations to execute
    */
    printf("------------------------------------------------------------\n");
    printf("For loop: size=%d\n", count);
    printf("------------------------------------------------------------\n");

    #pragma omp parallel 
    {
        #pragma omp for
        for(i = 0; i < count; i++)
            printf("Thread id %d printing value %d\n", omp_get_thread_num(), i);
    }

    // record end time
    end = omp_get_wtime();

    printf("Elapsed time: %f seconds\n", end - start);

    return 0;
}
