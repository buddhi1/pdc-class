#include <stdio.h>
#include <omp.h>

int main() 
{
	int i, j;
 
	/*
	 * nowait eliminates the implicit barrier after work sharing constructs
 	*/
	
	printf("Without nowait, loop i is always executed before loop j\n");
    #pragma omp parallel
	{
        #pragma omp for
		for(i = 0; i < 8; i++)
			printf("i is %d\n", i);
		
        #pragma omp for
		for(j = 0; j < 8; j++)
			printf("j is %d\n", j);
	}

	printf("With nowait, some iterations of loop i are executed after loop j\n");
    #pragma omp parallel
	{
        #pragma omp for nowait
		for(i = 0; i < 8; i++)
			printf("i is %d\n", i);
		
        #pragma omp for
		for(j = 0; j < 8; j++)
			printf("j is %d\n", j);
	}
	  
	return 0;
}
