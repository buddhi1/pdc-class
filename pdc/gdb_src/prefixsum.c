// Calculates prefix sum of a given array and prints the output
// Error: [i-2] should be replaced with [i-1]
#include <stdio.h>

void prefixsum(int arr[], int n) {
    int i;
    for (i=1; i<=n; i++) {
        arr[i]=arr[i]+arr[i-2];  // Intentional logical error
    }
}

void printArray(int arr[], int n) {
    int i;
    for (i=0; i<n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char* argv) {
    int arr[10]={1, 2, 3, 4, 5, 6, 7, 8, 9, 50};
    int n=10;

    printf("Array: ");
    printArray(arr, n);

    // Calculate prefix sum
    prefixsum(arr, n);

    // Print the prefix sum array
    printf("Prefix Sum Array: ");
    printArray(arr, n);

    return 0;
}
