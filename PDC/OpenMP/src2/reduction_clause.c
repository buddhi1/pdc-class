#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 0;
	int a[8] = {0,1,2,3, 0,1,2,3}; // 1*4 vector
	int b[8] = {4,5,6,7, 4,5,6,7}; // 4*1 vector
	int result = 0;

 
    /*
     * The following loop multiplies vectors a and b
     */
    #pragma omp parallel for reduction(+: result)
	for(i = 0; i < 8; i++){
		// printf("pid=%d Result=%d\n", omp_get_thread_num(), result);
		result += a[i]*b[i];  // 0+5+12+21=38
		// printf("pid=%d Result=%d\n", omp_get_thread_num(), result);
	}
	printf("Final result is %d\n", result);
	  
        return 0;
}
