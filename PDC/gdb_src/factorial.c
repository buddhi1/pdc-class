// User inputs a number and the factorial for that value is calculated
// Error: i<=n. '=' is missing
#include <stdio.h>

long factorial(int n) {
    long result=1;
    int i;
    for (i=1; i<n; i++) {
        result*=i;  
    }
    return result; 
}

int main(int argc, char* argv) {
    int num=5;
    // printf("Enter a number: ");
    // scanf("%d", &num);

    long fact = factorial(num);
    printf("Factorial of %d is: %ld\n", num, fact); // 5!=120 //4!=24

    return 0;
}
