#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 0;
	
	/*
	 * Threads simultaneously increment i by 1. Increment 
	 * requires three operations: 1) read in the old value of i;
	 * 2) add 1 to the old value; 3) write the new value back to i.
	 * 
	 * If there is no critical section, two threads may read in the same old
	 * value simultaneously and add 1 to the same old value and produce the
	 * same new value. That is, i is not incremented twice by two 
	 * threads, but incremented only once by two threads.
	 */
	printf("No critical section: i's final values are not stable\n");
    #pragma omp parallel
	{
                i = i + 1;
	}
	printf("Results (no critical section): i=%d\n", i);


	/*
	 * With critical section, each thread has to update i and j
	 * sequentially, so a thread is guaranteed to add 1 to i and j.
	 */
	i = 0;
	printf("With critical section: i's final values are stable "
			"(always the same as the number of threads).\n");
    #pragma omp parallel
	{
        #pragma omp atomic
	    i = i + 1;
	}
	printf("Results (with critical section): i=%d\n", i);
	
	return 0;
}
