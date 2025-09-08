#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 11;
	/*
	 * i is declared as private variable, and each thread is assigned with 
	 * its own instance of "i". Each thread accesses its own instance of "i"
	 * instead of the original one.
	 *
	 * Therefore, the value of "i" before and after the parallel construct
	 * will always be 0, with the values within the parallel construct are
	 * different.
	 *
	 */
	printf("Before parallel execution, i is %d\n", i);
    #pragma omp parallel private(i)
	{
		printf("Within a thread, i is %d\n", i);
	}

	printf("After parallel execution, i is %d\n", i);
	
	return 0;
}
