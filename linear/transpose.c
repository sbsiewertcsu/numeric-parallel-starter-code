// From https://www.geeksforgeeks.org/c-transpose-matrix/
//
// Adapt as necessary for transpose of any matrix including input from
// Lintest*.dat
//
// C Program to find transpose 
// of a square matrix 
#include <stdio.h> 
#define N 4 
  
// This function stores transpose 
// of A[][] in B[][] 
void transpose(int A[][N], int B[][N]) 
{ 
    int i, j; 
    for (i = 0; i < N; i++) 
        for (j = 0; j < N; j++) 
            // Assigns the transpose of element A[j][i] to 
            // B[i][j] 
            B[i][j] = A[j][i]; 
} 
  
// Driver code 
int main() 
{ 
    int A[N][N] = { { 1, 2, 3, 4 }, 
                    { 5, 6, 7, 8 }, 
                    { 9, 0, 1, 2 }, 
                    { 3, 4, 5, 6 } }; 
  
    int B[N][N], i, j; 
  
    transpose(A, B); 
  
    printf("Result matrix is \n"); 
    for (i = 0; i < N; i++) { 
        for (j = 0; j < N; j++) 
            printf("%d ", B[i][j]); 
        printf("\n"); 
    } 

    printf("Original matrix is \n"); 
    for (i = 0; i < N; i++) { 
        for (j = 0; j < N; j++) 
            printf("%d ", A[i][j]); 
        printf("\n"); 
    } 
  
    return 0; 
}

