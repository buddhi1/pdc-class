#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 0, j = 0;
 
    /*
     * Lastprivate variable is private. But the original variable's value is
	 * updated with the original value.
 	*/
    #pragma omp parallel for lastprivate(j)
	for(i = 0; i < 16; i++){
		j = i;
		printf("i=%d, j=%d\n", i,j);
	}
	printf("Final val of j is %d\n", j);
	  
    return 0;
}
