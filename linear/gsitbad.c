// https://www.codesansar.com/numerical-methods/gauss-seidel-iteration-using-c-programming.htm
//
// Modified by Sam Siewert 11/4/2021 to add more examples and to provide RHS verify.
// 
// Note that for GSIT, the equations must first be organized into diagonally dominant form,
// where the absolute value of the diagonal coefficient is greater than the sum of the absolute
// values of all other coefficients in that same row.  Otherwise, GSIT will not converge.
//
// Note that this example only works for dimension of 3, but could be generalized for any dimension.
// GSIT is normally used for large sparse matrices that are banded around the diagonal - i.e., most of
// the coefficients are on the diagonal or near it.  However, this program provides a nice simple
// example of how GSIT works.
//
// Check work and compare answers to https://www.symbolab.com/solver/system-of-equations-calculator
// Answers also can be checked with MATLAB.
//
// https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
//
// https://youtu.be/ajJD0Df5CsY
// https://youtu.be/lE3nKRpNOdw
//
#include<stdio.h>
#include<math.h>

// Note that this GSIT demonstration only works for specific dimension problems
#define DIM (3)

/* Arrange systems of linear
   equations to be solved in
   diagonally dominant form
   and form equation for each
   unknown and define here
*/
/* In this example we are solving
   3x + 20y - z = -18
   2x - 3y + 20z = 25
   20x + y - 2z = 17

   Symbolab Ans: x=1, y=-1, z=1
*/
/* Arranging given system of linear
   equations in diagonally dominant
   form:
   20x + y - 2z = 17
   3x + 20y -z = -18
   2x - 3y + 20z = 25
*/
/* Equations:
   x = (17-y+2z)/20
   y = (-18-3x+z)/20
   z = (25-2x+3y)/20
*/
/* Defining function */
//#define f1(x,y,z)  (17-y+2*z)/20
//#define f2(x,y,z)  (-18-3*x+z)/20
//#define f3(x,y,z)  (25-2*x+3*y)/20

// For verification, enter the coefficients into this array to match equations
//double a[DIM][DIM] = {{20.0, 1.0, -2.0}, {3.0, 20.0, -1.0}, {2.0, -3.0, 20.0}};
//double b[DIM] = {17.0, -18.0, 25.0};
//double x[DIM]={0.0, 0.0, 0.0};
//double sol[DIM]={1.0, -1.0, 1.0};
//int n = DIM;

/* In this alternate example we are solving
   3x + 2y + z = 7
   x + 7y + 10z = 17
   2x + 5y + 3z = 11

   Symbolab Ans: x=77/62, y=81/62, z=41/62
   Symbolab Ans: x=1.24193548387, y=1.30645161290, z=0.661290322580
*/
/* Arranging given system of linear
   equations in diagonally dominant
   form:
   3x + 2y + z = 7
   2x + 5y + 3z = 11
   x + 7y + 10z = 17
*/
/* Equations:
   x = (7-2y-z)/3
   y = (11-2x-3z)/5
   z = (17-x-7y)/10
*/
/* Defining function */
//#define f1(x,y,z)  (7-2*y-z)/3
//#define f2(x,y,z)  (11-2*x-3*z)/5
//#define f3(x,y,z)  (17-x-7*y)/10

// For verification, enter the coefficients into this array to match equations
//double a[DIM][DIM] = {{3.0, 2.0, 1.0}, {2.0, 5.0, 3.0}, {1.0, 7.0, 10.0}};
//double b[DIM] = {7.0, 11.0, 17.0};
//double x[DIM]={0.0, 0.0, 0.0};
//double sol[DIM]={1.24193548387, 1.30645161290, 0.661290322580};
//int n = DIM;


/* In this alternate example we are solving
   3x -0.1y -0.2z = 7.85
   0.1x + 7y + 0.3z = -19.3
   0.3x -0.2y + 10z = 71.4

   MATLAB Ans: x=3.0, y=-2.5, z=7.0
*/
/* Arranging given system of linear
   equations in diagonally dominant
   form - HERE WE ARE CREATING a PROBLEM by making NOT DD:
   Note that 20 is more than 6x 3
   Note that 30 is 3x 10

   3x -0.1y -20z = 7.85
   0.1x + 7y - 0.3z = -19.3
   30.0x -0.2y + 10z = 71.4
*/
/* Equations:
   x = (7.85 + 0.1y + 0.2z)/3
   y = (-19.3 -0.1x + 0.3z)/7
   z = (71.4 -30.0x + 0.2y)/10
*/
/* Defining function */
#define f1(x,y,z)  (7.85+0.1*y+20*z)/3
#define f2(x,y,z)  (-19.3-0.1*x+0.3*z)/7
#define f3(x,y,z)  (71.4-30.0*x+0.2*y)/10

/* Activity example with no apparent best diagonally dominate form 
3x + 7y  = 13
 x + 3y  =  7

 Adaptation of this code to different N left as an exercise for the
 reader.

*/

// For verification, enter the coefficients into this array to match equations
double a[DIM][DIM] = {{3.0, -0.1, -20}, {0.1, 7.0, -0.3}, {30.0, -0.2, 10.0}};
double b[DIM] = {7.85, -19.3, 71.4};
double x[DIM]={0.0, 0.0, 0.0};
int n = DIM;


void verify(double a[DIM][DIM], double *b, double *x, int n);
void vector_print(int nr, double *x);


int main(void)
{
    // Default guess is all are zero
    //
    double x0=0, y0=0, z0=0;

    // Better guess is that all are just diagonal-only solution
    //
    //double x0=(7.85/3.0), y0=(-19.3/7.0), z0=(71.4/10.0);

    double x1, y1, z1, e1, e2, e3, e;
    int count=1;

    printf("Enter tolerable error:\n");
    scanf("%lf", &e);

    printf("\nCount\tx\ty\tz\n");

    printf("%d\t%0.4f\t%0.4f\t%0.4f --- INITIAL GUESS\n", 0, x0,y0,z0);

    // if equations are arranged in diagonally dominate form and are not ill-conditioned, GSIT loop
    // should converge.
    //
    // For non-diaonally dominate inputs, GSIT will likely diverge (growing error).
    //
    do
    {
        /* Calculation */
        x1 = f1(x0,y0,z0);
        y1 = f2(x1,y0,z0);
        z1 = f3(x1,y1,z0);

        /* Error */
        e1 = fabs(x0-x1);
        e2 = fabs(y0-y1);
        e3 = fabs(z0-z1);

        printf("%d\t%0.4f\t%0.4f\t%0.4f, e1=%16.15lf, e2=%16.15lf, e3=%16.15lf\n", count, x1,y1,z1, e1, e2, e3);
        count++;

        /* Set value for next iteration */
        x0 = x1;
        y0 = y1;
        z0 = z1;

    //} while(e1>e && e2>e && e3>e);
    } while((e1>e) || (e2>e) || (e3>e));

    printf("\nGSIT Solution: x=%0.3f, y=%0.3f and z = %0.3f\n\n",x1,y1,z1);

    // Additional verification steps to assess error in answer.
    //
    x[0] = x1;
    x[1] = y1;
    x[2] = z1;

    verify(a, b, x, n);

    return 0;

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


///////////////////////////////////////////////////////////////////
// Multiply coefficient matrix "a" by solution vector s to see if
// it matches the expected RHS we started with.
//
//     a   - Matrix a[n][n]
//     b   - Right hand side vector b[n]
//     x   - Computed solution vector
//     n   - Matrix dimensions
////////////////////////////////////////////////////////////////////
void verify(double a[DIM][DIM], double *b, double *x, int n)
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

