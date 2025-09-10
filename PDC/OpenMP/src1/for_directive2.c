#include <stdio.h>
#include <omp.h>

int main() 
{
  int i, j, count1=8, count2=3;
  double start, end;

  // record start time
  start = omp_get_wtime();
  /*
   * create threads each takes some of the iterations to execution. Inner loop is local to each therad
  */
  printf("------------------------------------------------------------\n");
  printf("Nested for loop: outer loop size=%d, inner loop size=%d\n", count1, count2);
  printf("------------------------------------------------------------\n");

  #pragma omp parallel for private (j) 
	  for(i = 0; i < count1; i++)
      for (j = 0; j < count2; ++j)
        printf("i: %d, j: %d (pid = %d)\n", i, j, omp_get_thread_num());
  
  // record end time
  end = omp_get_wtime();

  printf("Elapsed time: %f seconds\n", end - start);

  return 0;
}
