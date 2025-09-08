#include <stdio.h>
#include <omp.h>

int main() 
{
  /*
   * create threads each will print "Hello world" and "Goodbye world"
   */
  int sum  = 0;
  #pragma omp parallel 

  {
    sum = sum + 1;
    printf("The current sum value is %d from thread %d\n", sum, omp_get_thread_num());
  }
  printf("###################################################");
  printf("The final sum value is %d  from thread %d\n", sum, omp_get_thread_num());
  return 0;
}
