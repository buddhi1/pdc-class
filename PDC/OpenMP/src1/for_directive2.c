#include <stdio.h>
#include <omp.h>

int main() 
{
  int i, j;
 
  /*
   * create threads each takes some of the iterations to execution. Inner loop is local to each therad
   */
  #pragma omp parallel for private (j) 
	  for(i = 0; i < 8; i++)
      for (j = 0; j < 3; ++j)
        printf("i: %d, j: %d\n", i, j);
  return 0;
}
