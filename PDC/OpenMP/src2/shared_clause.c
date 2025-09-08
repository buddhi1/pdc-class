#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 0;
	/*
	 * i is declared as shared variable. So all threads access the same 
	 * instance of "i."
	 *
	 * If "i" will be written by any of the threads, then accesses to "i"
	 * should be properly coordinated.
	 */
	printf("Before parallel execution, i is %d\n", i);
    #pragma omp parallel shared(i)
	{
		#pragma omp critical
		{
			i++;
			printf("Within a thread, i is %d\n", i);
		}
	}

	printf("After parallel execution, i is %d\n", i);
	
	return 0;
}
