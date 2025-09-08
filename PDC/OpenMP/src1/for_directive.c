#include <stdio.h>
#include <omp.h>

int main() 
{
  int i;
 
  /*
   * create threads each takes some of the iterations to execution
   */
  #pragma omp parallel 
  {
    #pragma omp for
	  for(i = 0; i < 16; i++)
		  printf("i is %d\n", i);
  }
	  
  return 0;
}
