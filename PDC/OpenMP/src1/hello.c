#include <stdio.h>
#include <omp.h>

int main() 
{
  /*
   * create threads each will print "Hello world" and "Goodbye world"
   */
  #pragma omp parallel
  {
    printf("Hello world ! \n");
    printf("Goodbye world ! \n");
  }

  return 0;
}
