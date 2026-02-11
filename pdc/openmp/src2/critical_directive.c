#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 0;
	int j = 0;
	
	/*
	 * Threads simultaneously increment i and j by 1 each. Increment 
	 * requires three operations: 1) read in the old value of i or j;
	 * 2) add 1 to the old value; 3) write the new value back to i or j.
	 * 
	 * If there is no critical section, two threads may read in the same old
	 * value simultaneously and add 1 to the same old value and produce the
	 * same new value. That is, i or j is not incremented twice by two 
	 * threads, but incremented only once by two threads.
	 */
	printf("No critical section: i, j's final values are not stable\n");
    #pragma omp parallel
	{
        i = i + 1;
		j = j + 1;
	}
	printf("Results (no critical section): i=%d, j=%d\n", i,j);


	/*
	 * With critical section, each thread has to update i and j
	 * sequentially, so a thread is guaranteed to add 1 to i and j.
	 */
	i = j =0;
	printf("With critical section: i, j's final values are stable "
			"(always the same as the number of threads).\n");
    #pragma omp parallel
	{
        #pragma omp critical
		{
		    i = i + 1;
		    j = j + 1;
        }
	}
	printf("Results (with critical section): i=%d, j=%d\n", i,j);
	
	return 0;
}
