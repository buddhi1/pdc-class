#include <stdio.h>
#include <omp.h>

int main() 
{
	int i;
	int id;
	
	/*
	 * omp_get_thread_num return the number (id) of the calling thread
 	*/
    #pragma omp parallel private(id)
	{
		id = omp_get_thread_num();
		#pragma omp for
		for(i = 0; i < 16; i++)
			printf("thread %d handles iteration i = %d\n", id, i);
	}
	  
	return 0;
}
