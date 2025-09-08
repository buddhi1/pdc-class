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
				printf("i is %d\n", i);
			
            #pragma omp section
			for(j = 0; j < ITERS; j++)
				printf("j is %d\n", j);
		}
	}
	  
  return 0;
}
