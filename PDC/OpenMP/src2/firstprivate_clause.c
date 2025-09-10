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
	printf("Before parallel execution, i is %d (pid = %d)\n", i, omp_get_thread_num());
	#pragma omp parallel firstprivate(i)
	{
		printf("Within a pid=%d: initial value is %d\n", omp_get_thread_num(), i);
		i++;
		printf("Within a pid=%d: new value is %d\n", omp_get_thread_num(), i);
	}

	printf("After parallel execution, i is %d (pid = %d)\n", i, omp_get_thread_num());
	
	return 0;
}