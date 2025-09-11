#include <stdio.h>
#include <omp.h>

int main() 
{
	/*
	 * create threads each will print "Hello world" and "Goodbye world"
 	*/
    #pragma omp parallel
	{		
		printf("Hello world\n");
		#pragma omp master
		{
			printf("I am the master! (printed by pid %d)\n", omp_get_thread_num());
		}
		printf("Goodbye world\n");
	}
	
	return 0;
}
