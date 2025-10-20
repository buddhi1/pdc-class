// Devide a number by a denominator
// Error: Divide by 0 is not handled
#include <stdio.h>

float divide(float numerator, float denominator) {
    // if (denominator == 0) {
    //     printf("Error: Division by zero is not allowed.\n");
    //     return 0;  // You can choose to handle this error differently
    // }
    return numerator/denominator;
}

int main(int argc, char* argv) {
    float result1, result2;

    // First division: 5 / 3
    result1 = divide(5.0, 3.0);
    printf("Result of 5 / 3: %.2f\n", result1);

    // Second division: 4 / 0
    result2 = divide(4.0, 0.0);
    printf("Result of 4 / 0: %.2f\n", result2);

    return 0;
}
