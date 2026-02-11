#include <stdio.h>
#include <omp.h>

int main() 
{
	/*
	 * create threads each will print "Hello world" and "Goodbye world"
	 * barrier directive specifies that all "Hello world" be printed before
	 * "Goodbye world"
	 */
    #pragma omp parallel
	{
		printf("Hello world\n");
        #pragma omp barrier
		printf("Goodbye world\n");
	}
	
	return 0;
}
