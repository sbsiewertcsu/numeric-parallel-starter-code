/////////////////////////////////////////////////////////////////////////////////////
//
// Original code from https://web.mit.edu/10.001/Web/Course_Notes/Gauss_Pivoting.c
//
// Updated for readability and tracing by Sam Siewert, 4/13/2021 with following:
//
// 1) Remove index starting at "1" to better confrom to C programming style
// 2) Update index range for PP factor subtraction to allow for tracing of zeros below upper form "U"
// 3) Modernization of style to conform with C programming for comments, basic blocks, etc.
// 4) Addition of a verification function to be used after back-solve step
// 5) More clear and concise commenting for GEWPP main steps of:
//    i)   PP factor and row determination
//    ii)  Row swaps if needed
//    iii) Update of coefficient matrix "a" including computed ZEROs
//    iv)  Back-solve
//    v)   Use of solution "x" to verify by multiplication with "a", which should match RHS "b"
//
// For more background see:
//
// https://web.mit.edu/10.001/Web/Course_Notes/GaussElimPivoting.html
// https://en.wikipedia.org/wiki/Gaussian_elimination
//
// For great videos on GEWPP, see these Youtube videos on the process
// https://www.youtube.com/watch?v=Yl_I4iYlyus
// https://www.youtube.com/watch?v=6p-a7ZoGk18&t=38s
// https://www.youtube.com/watch?v=_q5HpaOhFes&t=26s
//   
// ________________________________________________________________
// 
// Name:      Gauss_Pivot.c
// 
// Purpose:   This program implements Gaussian Elimination
//            with pivoting. The program will read multiple sets
// 		      of problems and print the answers.
// 			
// Author(s): R. Sureshkumar (10 January 1997)
//            Gregory J. McRae (22 October 1997)
// 
// Address:   Department of Chemical Engineering
//            Room 66-372
//            Massachusetts Institute of Technology
//            Cambridge, MA 02139
// 		      mcrae@mit.edu
//
// Refactoring:
//            Sam Siewert (13 April 2021)
//
// Address:   Department of Computer Science
//            O'Connell Technology Center, 211
//            Chico, CA 95929-0410
//            sbiewert@csuchico.edu
// 
// Usage:     The program expects a data file 'gauss.dat' default or similar file that is 
//            structured as follows:
// 		   
// 		      Line 1:   A brief description (< 80 characters)
// 		      Line 2:   The dimension of the problem (n >= 1)
// 		      Line 3+:  Row 1 to n of the matrix A
// 		      Line n+3: Row 1 to n of the matrix b
// 
// Example format of "gauss.dat"
// 4
//  2.  1.  0.  0
//  1.  2.  1.  0
//  0.  1.  2.  1
//  0.  0.  1.  2
// 2.  
// 1.  
// 4.  
// 8.
// ________________________________________________________________		    
//
// Examples tested include:
//
// #1 - gauss.dat default from Strang (Applied Mathematics, p.10)  Solution = x = {1, 0, 0, 4}  
// #2 - Lintest1.dat test input from CSCI 551 class at CSU
// #3 - Other?
//
////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ICHAR 80  // Length of array holding description of the problem
#define IDEBUG 1  // Flag to enable printing of intermediate results of decomposition =1 yes, =0 no.


// Function Prototypes
void  matrix_print (int nr, int nc, double **A);
void  vector_print (int nr, double *x);
void gauss(double **a, double *b, double *x, int n);
void verify(double **a, double *b, double *x, int n);


int main (int argc, char *argv[])
{
    double  **a, *b, *x; // a=coefficients, b=RHS, x=solution vector
    char   desc[ICHAR];  // file header description 
    int    i, j, n;  
    FILE   *finput;    

    // Open the file containing a description, n, A and b
    //
    // Use default gauss.dat or if arguments supplied, use input file with specific test.
    //
    if(argc > 1)
    {
       printf("Using custom input file %s: argc=%d, argv]0]=%s, argv[1]=%s\n", argv[0], argc, argv[0], argv[1]);
       finput = fopen(argv[1],"r");
    }
    else
    {
        printf("Using gauss.dat DEFAULT example input file\n");
        finput = fopen("gauss.dat","r");
    }

    if (finput == NULL) 
    { 
    	printf("Data file gauss.dat not found\n");
    	return(-1);
    }

    // Get a one line description of the matrix problem and
    // then the dimension, n, of the system A[n][n] and b[n]  
    fgets(desc, ICHAR , finput);      
    fscanf(finput, "%d", &n);
    printf("%s", desc);  
    printf("\nDimension of matrix = %d\n\n", n);

    // Dynamic allocation of the arrays and vectors
    //
    a = calloc(n, sizeof(double *));

    for(i = 0; i<n; ++i) 
    { 
    	a[i] =  calloc(n, sizeof(double));
    }

    b = calloc(n, sizeof(double));
    x = calloc(n, sizeof(double));

    printf("Memory allocation done\n");

    // Read the elements of A
    for (i=0; i<n; i++)
    {
    	for (j=0; j<n; j++) 
        {
    		fscanf(finput,"%lf ",&a[i][j]);
    	}
    }

    printf("Coefficient array read done\n");

    // Read the elements of b
    for (i=0; i<n; i++)
    {
       fscanf(finput,"%lf ",&b[i]);
    }
       
    printf("RHS vector read done\n");

    fclose(finput); // Close the input file

    printf("\nMatrices read from input file\n");

    printf("\nMatrix A\n\n");
    matrix_print(n, n, a);

    printf("\nVector b\n\n");
    vector_print(n, b);

    // Call the Gaussian elimination function
	gauss(a, b, x, n); 

    printf("\nSolution x\n\n");
    vector_print(n, x);

    // Multiply solution "x" by matrix "a" to verify we get RHS "b"
	verify(a, b, x, n); 

    return(0);
} 


///////////////////////////////////////////////////////////////////
// Multiply matrix a X solution vector s to see if it matches rhs
//
//	   a   - Matrix a[n][n]
//	   b   - Right hand side vector b[n]
//	   x   - Computed solution vector
//	   n   - Matrix dimensions
////////////////////////////////////////////////////////////////////
void verify(double **a, double *b, double *x, int n)
{
    int i, j;
    double rhs[n];

    // for all rows
    for (i=0; i<n; ++i) 
    {
        rhs[i] = 0.0;

        // sum up row's column coefficient x solution vector element
        for (j=0; j<n; ++j)
        {
            rhs[i] += a[i][j] * x[j];
        }
    }

    // Compare original RHS "b" to computed RHS
    printf("Computed RHS is:\n");
    vector_print(n, rhs);
    printf("Original RHS is:\n");
    vector_print(n, b);
}


/////////////////////////////////////////////////////////
//
// Name:      gauss
//
// Purpose:   This program uses Gaussian Elimination with
//            with pivoting to solve the problem A x =b.
//			
// Author(s): R. Sureshkumar (10 January 1997)
//            Modified by: Gregory J. McRae (22 October 1997)
//
// Address:   Department of Chemical Engineering
//            Room 66-372
//            Massachusetts Institute of Technology
//            Cambridge, MA 02139
//		      mcrae@mit.edu
//
// Refactoring:
//            Sam Siewert (13 April 2021)
//
// Address:   Department of Computer Science
//            O'Connell Technology Center, 211
//            Chico, CA 95929-0410
//            sbiewert@csuchico.edu
//
// Usage:     gauss(a,b,x,n)
//		   
//		   a   - Matrix a[n][n] of coefficients
//		   b   - Right hand side vector b[n]
//		   x   - Desired solution vector
//		   n   - Matrix dimensions
//
//           If IDEBUG is set to 1 the program will print
//           out the intermediate decompositions as well
//           as the number of row interchanges.
///////////////////////////////////////////////////////////
 
void gauss(double **a, double *b, double *x, int n)
{
    int   i, j, k, m, rowx;
    double xfac, temp, temp1, amax;

    printf("\nMatrix A passed in\n");
    matrix_print(n, n, a);

    //////////////////////////////////////////
    //
    //  Do the forward reduction step. 
    //////////////////////////////////////////

    // Keep count of the row interchanges
    rowx = 0;   

    for (k=0; k<(n-1); ++k) 
    {

         amax = (double) fabs(a[k][k]) ;
         m = k;

         // Find the row with largest pivot
         for (i=k+1; i<n; i++)
         {   
             xfac = (double) fabs(a[i][k]);

             if(xfac > amax) 
             {
                 amax = xfac; m=i;
             }
         }
         printf("\nPivot row=%d\n", m);


         // Row interchanges for partial pivot
         if(m != k) 
         {  
             printf("Row swaps with m=%d, k=%d\n", m, k);

             rowx = rowx+1;
             temp1 = b[k];
             b[k]  = b[m];
             b[m]  = temp1;

             for(j=k; j<n; j++) 
             {
                 temp = a[k][j];
                 a[k][j] = a[m][j];
                 a[m][j] = temp;
             }

             printf("\nMatrix A after row swaps\n");
             matrix_print(n, n, a);
          }

          // Row scaling to get zero
          for (i=k+1; i<n; ++i) 
          {
              xfac = a[i][k] / a[k][k];

              // Original solution did not inclue ZERO columns, assumed to be zero
              //
              // We add the ZEROs back and trace computed to help understand round-off error that can
              // occurr with GEWPP.  All of the n-1 row column elements below the pivot row should be zero by
              // computation, but might have some error, so we want to see it when we print out the intermediate
              // matrices.
              //
              for (j=k; j<n; ++j) 
              {
                  a[i][j] = a[i][j] - (xfac*a[k][j]);
              }
              
              b[i] = b[i] - (xfac*b[k]);

              printf("\nMatrix A after row scaling with xfac=%lf\n", xfac);
              matrix_print(n, n, a);
          }

        if(IDEBUG == 1) 
        {
            printf("\n A after decomposition step %d\n\n", k+1);
            matrix_print(n, n, a);
        }        

    }

    ////////////////////////////////////////
    //
    // Do the back substitution step 
    //
    ////////////////////////////////////////
    for (j=0; j<n; ++j) 
    {

      // Start at last row and work upward to first
      k=n-j-1;

      x[k] = b[k];

      for(i=k+1; i<n; ++i) 
      {
         x[k] = x[k] - (a[k][i]*x[i]);
      }

      x[k] = x[k] / a[k][k];
    }

    if(IDEBUG == 1) 
        printf("\nNumber of row exchanges = %d\n",rowx);
}


//////////////////////////////////////////////////////////////////////
//
// Name:     matrix_print
//
// Purpose:  This function will print out a general two-dimensional 
//           matrix given supplied dimensions.
//			
// Usage:    mat_print(nr, nc, A);
//
// Input:    nr     - number of rows (must be >= 1)
// 		     nc     - number of columns (must be >= 1)
// 		     A      - Matrix A[nr][nc] to be printed
//
//////////////////////////////////////////////////////////////////////
void matrix_print(int nr, int nc, double **A)
{
    int i, j;
  
    for (i = 0; i < nr; i++) 
    {
     	for (j = 0; j < nc; j++) 
        {
	    	printf ("%9.4f  ", A[i][j]);
	    }

	    printf("\n"); // Insert a new line at end of each row
    }
}



///////////////////////////////////////////////////////////////////
// Name:     vector_print
// 
// Purpose:  This function will print out a one-dimensional 
//           vector with supplied dimension.
// 			
// Usage:    vector_print(nr, x);
//
// Input:    nr     - number of rows (must be >= 1)
//           x      - Vector x[nr] to be printed
//
///////////////////////////////////////////////////////////////////
void vector_print(int nr, double *x)
{
    int i;
  
    for (i = 0; i < nr; i++) 
    {
    	printf ("%9.4f  \n", x[i]);
    }

    printf("\n");  // Insert a new line at the end
}
