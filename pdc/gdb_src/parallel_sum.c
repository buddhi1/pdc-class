// Calculates sum of values in an array in parallel
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 5
#define _DEBUG_

int main(int argc, char* argv) {
    int sum=0;
    int i;

    // Parallel for loop with an error
    #pragma omp parallel for reduction(+:sum)
    for (i=0; i<=N; i++) {  // Introduce an error by using "<=" instead of "<"
        // Each iteration of this loop can be executed in parallel
        #ifdef _DEBUG_
            printf("Thread %d is working on iteration %d\n", omp_get_thread_num(), i);
        #endif
        sum+=i;
    }

    printf("Sum of %d is %d\n", N, sum);

    return 0;
}
