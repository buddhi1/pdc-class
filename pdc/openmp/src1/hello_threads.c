#include <stdio.h>
#include <omp.h>

int main() 
{
  /*
   * create threads each will print "Hello world" and "Goodbye world"
   */
  #pragma omp parallel
  {  
    printf("Hello world from thread %d\n", omp_get_thread_num());
    printf("Goodbye world from thread %d\n", omp_get_thread_num());
  }

  return 0;
}
