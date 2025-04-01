/////////////////////////////////////////////////////////////////////////////////////
//
// Original code from https://web.mit.edu/10.001/Web/Course_Notes/Gauss_Pivoting.c
//
// Updated for readability and tracing by Sam Siewert, 4/13/2021 and 10/27/21 with following:
//
// 1) Remove index starting at "1" to better confrom to C programming style
// 2) Update index range for PP factor subtraction to allow for tracing of zeros below diagonal
// 3) Modernization of style to conform with C programming for comments, basic blocks, etc.
// 4) Addition of a verification function to be used after back-solve step
// 5) More clear and concise commenting for GEWPP main steps of:
//    i)   PP factor and row determination
//    ii)  Row swaps if needed
//    iii) Update of coefficient matrix "a" including computed ZEROs
//    iv)  Back-solve
//    v)   Use of solution "x" to verify by multiplication with "a", which should match RHS "b"
// 6) Improved variable names for understanding
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
// https://www.youtube.com/watch?v=euIXYdyjlqo (above videos all in one)
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
#define TRACE 1 

// Function Prototypes
void  matrix_print (int nr, int nc, double **A);
void  vector_print (int nr, double *x);
void  augmented_matrix_print (int nr, int nc, double **A, double *x);
void gauss(double **a, double *b, double *x, int n);
void verify(double **a, double *b, double *x, int n);


int main (int argc, char *argv[])
{
    double  **a, *b, *x;        // a=coefficients, b=RHS, x=solution vector
    double  **asave, *bsave;    // asave=coefficients original, bsave=RHS original
    char   desc[ICHAR];         // file header description 
    int    row_idx, col_jdx, n; // indexes into arrays and dimention of linear system
    FILE   *finput;             // file pointer to description, dimension, coefficients, and RHS

    // Open the file containing:
    // 1) a description, 
    // 2) n = dimensions (n equations, n unknowns),
    // 3) A = matrix of coefficients for LHS, and
    // 4) b = RHS
    //
    // Use default gauss.dat or if arguments supplied, use input file with specific test.
    //
    // x is the solution vector (unknowns) that we are solving for
    //
    if(argc > 1)
    {
       printf("Using custom input file %s: argc=%d, argv[0]=%s, argv[1]=%s\n",
              argv[0], argc, argv[0], argv[1]);
       finput = fopen(argv[1],"r");
    }
    else
    {
        printf("Using gauss.dat DEFAULT example input file\n");
        printf("Use: gewpp [Lintest#.dat]\n");
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
    // Allocate n pointers that point to each row in a 2D array
    //
    a = calloc(n, sizeof(double *));
    asave = calloc(n, sizeof(double *));

    // Now allocate n rows with n columns each, one row at a time
    //
    // A 2D array could alternatively be used, but this approach allows us
    // to use C pointers, for better or worse, which are efficient.
    //
    // possible loop pragma here
    //
    for(row_idx = 0; row_idx < n; ++row_idx) 
    { 
        // Allocate n column values for this row
    	a[row_idx] =  calloc(n, sizeof(double));
    	asave[row_idx] =  calloc(n, sizeof(double));
    }

    // Result of above is a 2D coefficient array "a", to contain coefficents a[0][0] ... a[n-1][n-1], 
    // which will be transformed via GEWPP into the lower diagonal form with n-1 zeros in the first
    // column and n-1 zeros in the last row, such that we have the following for a 4x4:
    //
    // a[0][0], ...    , ...,     a[0][3]
    // 0      , a[1][1], ...,     a[1][3]
    // 0      , 0      , a[2][2], a[2][3]
    // 0      , 0      , 0  ,     a[3][3]
    //
    // All zeros below the diagonal.
    //

    // Allocate space for the RHS vector of dimension n
    b = calloc(n, sizeof(double));
    bsave = calloc(n, sizeof(double));

    // Allocate space for the solution vector (unknowns) of dimension n
    x = calloc(n, sizeof(double));

    printf("Memory allocation done\n");

    // Read the elements of coefficient array A
    for (row_idx=0; row_idx < n; row_idx++)
    {
    	for (col_jdx=0; col_jdx < n; col_jdx++) 
        {
    		fscanf(finput,"%lf ",&a[row_idx][col_jdx]);
            asave[row_idx][col_jdx]=a[row_idx][col_jdx];
    	}
    }

    printf("Coefficient array read done\n");

    // Read the elements of RHS vector "b"
    for (row_idx=0; row_idx < n; row_idx++)
    {
       fscanf(finput,"%lf ",&b[row_idx]);
       // RHS b gets rearranged by pivoting, so save original
       bsave[row_idx]=b[row_idx];
    }
       
    printf("RHS vector read done\n");

    fclose(finput); // Close the input file

    if(TRACE)
    {
        // Now print out the problem to be solved in vector matrix form for n equations, n unknowns
        printf("\nMatrices read from input file\n");
        printf("\nMatrix A passed in\n");
        matrix_print(n, n, a);

        printf("\nRHS Vector b\n\n");
        vector_print(n, b);

        printf("\nAugmented Coefficient Matrix A with RHS\n");
        augmented_matrix_print(n, n, a, b);
    }

    // Call the Gaussian elimination function with back-solve
    //
    // This function uses row re-arrangement, scaling, and row operations
    // in a systematic way to get the lower zero form or "L" form of the coefficient matrix
    // and RHS where all coefficients below the diagonal are zero.
    //
    // Note that this function call will potentially rearrange "a" and "b", so we must save a and b
    // from original input for verification.
    //
	gauss(a, b, x, n); 

    printf("\nSolution x\n\n");
    vector_print(n, x);

    // Multiply solution "x" by matrix "a" to verify we get RHS "bsave"
	verify(asave, bsave, x, n); 

    return(0);
} 


///////////////////////////////////////////////////////////////////
// Multiply coefficient matrix "a" by solution vector s to see if
// it matches the expected RHS we started with.
//
//	   a   - Matrix a[n][n]
//	   b   - Right hand side vector b[n]
//	   x   - Computed solution vector
//	   n   - Matrix dimensions
////////////////////////////////////////////////////////////////////
void verify(double **a, double *b, double *x, int n)
{
    int row_idx, col_jdx;
    double rhs[n];

    // for all rows
    for (row_idx=0; row_idx < n; ++row_idx) 
    {
        rhs[row_idx] = 0.0;

        // sum up row's column coefficient x solution vector element
        // as we would do for any matrix * vector operation which yields a vector,
        // which should be the RHS
        for (col_jdx=0; col_jdx < n; ++col_jdx)
        {
            rhs[row_idx] += a[row_idx][col_jdx] * x[col_jdx];
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
//            with pivoting to solve the problem A x = b,
//            where A is a matrix, x a vector of unknowns, 
//            and b the RHS for the vector/matrix form of n
//            equations with n unknowns.
//
// Method:    Gaussian Elimination with Partial Pivoting
//            1) Forward reduction to get zeros in lower diagonal (finding diagonal pivot)
//            2) Back solve using last row, last column coefficient (diagonal)
//               and RHS, repeating to find all remaining unknowns
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
    int   row_idx, col_jdx, coef_idx, search_idx, pivot_row, solve_idx, rowx;
    double xfac, temp, amax;

    if(TRACE)
    {
        printf("\nAugmented Coefficient Matrix A with RHS passed to GEWPP\n");
        augmented_matrix_print(n, n, a, b);
    }

    //////////////////////////////////////////
    //
    //  Do the forward reduction step. 
    //////////////////////////////////////////

    // Keep count of the row interchanges
    rowx = 0;   

    for (search_idx=0; search_idx < (n-1); ++search_idx) 
    {

         // Assume first row, first column coefficient is largest to start the
         // search for the true largest coefficient in the matrix "a"
         //
         amax = (double) fabs(a[search_idx][search_idx]);
         pivot_row = search_idx; // assume first row is the pivot row to start

         // Find the row with largest pivot (coefficient)
         //
         for (row_idx=search_idx+1; row_idx < n; row_idx++)
         {   
             xfac = (double) fabs(a[row_idx][search_idx]);

             if(xfac > amax) 
             {
                 amax = xfac; pivot_row=row_idx;
             }
         }
         if(TRACE) printf("\nPivot row=%d, Pivot coeff=%lf, Pivot factor=%lf\n", pivot_row, amax, xfac);


         // Row interchanges for partial pivot to get upper diagonal form
         if(pivot_row != search_idx) 
         {  
             printf("Row swaps with pivot_row=%d, search_idx=%d\n", pivot_row, search_idx);

             rowx = rowx+1;
             temp = b[search_idx];
             b[search_idx]  = b[pivot_row];
             b[pivot_row]  = temp;

             for(col_jdx=search_idx; col_jdx < n; col_jdx++) 
             {
                 temp = a[search_idx][col_jdx];
                 a[search_idx][col_jdx] = a[pivot_row][col_jdx];
                 a[pivot_row][col_jdx] = temp;
             }

             if(TRACE)
             {
                 printf("\nAugmented Matrix A with RHS after row swaps\n");
                 augmented_matrix_print(n, n, a, b);
             }
          }

          // Row scaling to get zero in corresponding column
          for (row_idx=search_idx+1; row_idx < n; ++row_idx) 
          {
              xfac = a[row_idx][search_idx] / a[search_idx][search_idx];

              if(TRACE) printf("coeff1=%lf, coeff2=%lf, xfac=%lf\n", a[row_idx][search_idx], a[search_idx][search_idx], xfac);

              // Original solution from MIT did not inclue ZERO columns, and they are
              // assumed to be zero as was noted in the GEWPP tutorial videos.
              //
              // We add the ZEROs back and trace all computed values to help understand round-off
              // error that can occurr with GEWPP.  All of the n-1 row column elements below the pivot
              // row should be zero by computation, but might have some error, so we want to see
              // it when we print out the intermediate matrices, if in fact there is error.
              //
              for (col_jdx=search_idx; col_jdx < n; ++col_jdx) 
              {
                  a[row_idx][col_jdx] = a[row_idx][col_jdx] - (xfac*a[search_idx][col_jdx]);

                  if(TRACE) printf("coeff for zero=%lf, coeff mult by xfac = %lf\n", a[row_idx][col_jdx], (xfac*a[search_idx][col_jdx]));
              }
              
              b[row_idx] = b[row_idx] - (xfac*b[search_idx]);

              if(TRACE) printf("\ncoeff for RHS=%lf, coeff mult by xfac = %lf\n", b[row_idx], (xfac*b[search_idx]));

              if(TRACE)
              {
                  printf("\nAugmented Matrix A with RHS after row scaling with xfac=%lf\n", xfac);
                  augmented_matrix_print(n, n, a, b);
              }
          }

        if(IDEBUG == 1) 
        {
            printf("\nAugmented Matrix A with RHS after lower diagonal decomposition step %d\n\n", search_idx+1);
            augmented_matrix_print(n, n, a, b);
        }        

    }

    ////////////////////////////////////////
    //
    // Do the back substitution step 
    //
    ////////////////////////////////////////
    for (row_idx=0; row_idx < n; ++row_idx) 
    {

      // Start at last row and work upward to first row
      //
      // The last row should always just have one non-zero coefficient in the last
      // column.  After solving for this unknown in the last row, we can then use it
      // to solve for the unknown one row up, and so on.
      solve_idx=n-row_idx-1;

      // Start out with solution as RHS
      x[solve_idx] = b[solve_idx];

      // Note that this loop is skipped for the first solution which is simply
      // the RHS / (last row, last column coefficient), or RHS / diagonal[last][last]
      //
      // In subsequent rows as we move up, the result from the prior solution row is used
      // to determine the current.  E.g., for 3 unknowns x, y, z, this automates finding
      // z first, then using z to find y, and finally using y and z to find x.
      for(coef_idx=solve_idx+1; coef_idx < n; ++coef_idx) 
      {
         x[solve_idx] = x[solve_idx] - (a[solve_idx][coef_idx]*x[coef_idx]);
      }

      // based on lower diagonal form we always divide by a diagonal coefficient to
      // find the current unknown of interest
      x[solve_idx] = x[solve_idx] / a[solve_idx][solve_idx];

      if(TRACE) printf("x[%d]=%lf\n", solve_idx, x[solve_idx]);
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
    int row_idx, col_jdx;
  
    for (row_idx = 0; row_idx < nr; row_idx++) 
    {
     	for (col_jdx = 0; col_jdx < nc; col_jdx++) 
        {
	    	printf ("%9.4f  ", A[row_idx][col_jdx]);
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
    int row_idx;
  
    for (row_idx = 0; row_idx < nr; row_idx++) 
    {
    	printf ("%9.4f  \n", x[row_idx]);
    }

    printf("\n");  // Insert a new line at the end
}


//////////////////////////////////////////////////////////////////////
//
// Name:     augmented_matrix_print
//
// Purpose:  This function will print out a general two-dimensional 
//           matrix with RHS as augmented matrix given supplied dimensions.
//			
// Usage:    mat_print(nr, nc, A);
//
// Input:    nr     - number of rows (must be >= 1)
// 		     nc     - number of columns (must be >= 1)
// 		     A      - Matrix A[nr][nc] to be printed
//           x      - Vector x[nr] to be printed
//
//////////////////////////////////////////////////////////////////////
void augmented_matrix_print(int nr, int nc, double **A, double *x)
{
    int row_idx, col_jdx;
  
    for (row_idx = 0; row_idx < nr; row_idx++) 
    {
     	for (col_jdx = 0; col_jdx < nc; col_jdx++) 
        {
	    	printf ("%9.4f ", A[row_idx][col_jdx]);
	    }
        printf("%9.4f", x[row_idx]);
	    printf("\n"); // Insert a new line at end of each row
    }
}

