#include <stdio.h>
#include <omp.h>

int main() 
{
	/*
	 * create threads each will print "Hello world" and "Goodbye world"
 	*/
    #pragma omp parallel
	{
		#pragma omp master
		{
			printf("I am the master! (printed by %d)\n", omp_get_thread_num());
		}
		
		printf("Hello world\n");
		printf("Goodbye world\n");
	}
	
	return 0;
}
