#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 3;
	/*
	 * i is declared as private variable, and each thread is assigned with 
	 * its own instance of "i". Each thread accesses its own instance of "i"
	 * instead of the original one.
	 *
	 * However, unlike PRIVATE, FIRSTPRIVATE variables are initialized using
	 * the values of the original instances.
	 *
	 */
	printf("Before parallel execution, i is %d\n", i);
    #pragma omp parallel firstprivate(i)
	{
		printf("Within a thread, i's init value is %d\n", i);
		i++;
		printf("Within a thread, i's new value is %d\n", i);
	}

	printf("After parallel execution, i is %d\n", i);
	
	return 0;
}
