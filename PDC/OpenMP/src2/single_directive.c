#include <stdio.h>
#include <omp.h>

int main() 
{
	/*
	 * create threads each will print "Hello world" and "Goodbye world"
 	*/
    #pragma omp parallel
	{
		#pragma omp single
		{
			printf("This only prints once (printed by pid %d)\n", omp_get_thread_num());
		}
		printf("Hello world\n");
		printf("Goodbye world\n");
		
	}
	
	return 0;
}
