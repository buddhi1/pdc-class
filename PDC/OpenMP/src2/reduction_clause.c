#include <stdio.h>
#include <omp.h>

int main() 
{
	int i = 0;
	int a[4] = {0,1,2,3}; // 1*4 vector
	int b[4] = {4,5,6,7}; // 4*1 vector
	int result = 0;
 
    /*
     * The following loop multiplies vectors a and b
     */
    #pragma omp parallel for reduction(+: result)
	for(i = 0; i < 4; i++){
		result += a[i]*b[i];
	}
	printf("Final result is %d\n", result);
	  
        return 0;
}
