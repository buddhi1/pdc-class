#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 0, j = 0;
 
    /*
     * Lastprivate variable is private. But the original variable's value is
	 * updated with the original value.
 	*/
	printf("Before parallel execution, i=%d and j=%d (pid = %d)\n", i, j, omp_get_thread_num());

    #pragma omp parallel for lastprivate(j)
	for(i = 0; i < 16; i++){
		j = i;
		printf("pid=%d i=%d, j=%d\n", omp_get_thread_num(), i,j);
	}
	printf("After parallel execution, i=%d and j=%d (pid = %d)\n", i, j, omp_get_thread_num());
	  
    return 0;
}