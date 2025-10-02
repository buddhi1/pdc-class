#include <stdio.h>
#include <omp.h>
#define N 128

int main() 
{
	int i = 0;
	int a[N];
	int b[N];
	int result = 0;

	for(i = 0; i < N; ++i){
		a[i]=1;
		b[i]=2;
	}
 
    /*
     * The following loop multiplies vectors a and b
     */
    #pragma omp parallel for reduction(+: result)
	for(i = 0; i < N; i++){
		// printf("pid=%d Result=%d\n", omp_get_thread_num(), result);
		result += a[i]*b[i];  
		// printf("pid=%d Result=%d\n", omp_get_thread_num(), result);
	}
	printf("Final result is %d\n", result);
	  
        return 0;
}
