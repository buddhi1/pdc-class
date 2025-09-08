#include <stdio.h>
#include <omp.h>

int main() 
{
  /*
   * create threads each will print "Hello world" and "Goodbye world"
   */
  int id;
  #pragma omp parallel private(id)

  {
    id = omp_get_thread_num();
   
    printf("Hello world from thread %d\n", id);
    #pragma omp barrier
    printf("Goodbye world from thread %d\n", id);
  }

  return 0;
}
