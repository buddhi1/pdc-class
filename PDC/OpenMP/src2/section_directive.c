#include <stdio.h>
#include <omp.h>

#define ITERS 8

int main() 
{
	int i,j;
 
	/*
	 * create threads each takes some of the iterations to execution
	 */
    #pragma omp parallel
	{
        #pragma omp sections
		{
            #pragma omp section
			for(i = 0; i < ITERS; i++)
				printf("i is %d (pid=%d)\n", i, omp_get_thread_num());
			
            #pragma omp section
			for(j = 0; j < ITERS; j++)
				printf("j is %d (pid=%d)\n", j, omp_get_thread_num());
		}
	}
	  
  return 0;
}
