#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//#define MAX_DIM (10000) // can cause a linking error, e.g., on ecc-linux

#define MAX_DIM (1000)
#define DEFAULT_DIM (1000)

// Function Prototypes
void  matrix_print (int nr, int nc, double A[][MAX_DIM]);
void  vector_print (int nr, double x[MAX_DIM]);
void vmult(int n, double A[][MAX_DIM], double x[MAX_DIM]);
void mmult(int n, double A[][MAX_DIM], double B[][MAX_DIM]);

double default_mat[MAX_DIM][MAX_DIM];
double alt_mat[MAX_DIM][MAX_DIM];
double default_vect[MAX_DIM];
double C[MAX_DIM][MAX_DIM];

int debug=0; // debug is normally off, but turned on for simple examples

// global for simple demonstration of various pragma locations
int thread_count=4;

double notesA[3][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}};
double notesB[3][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}};
double notesC[3][3] = {{15.0, 18.0, 21.0}, {42.0, 54.0, 66.0}, {69.0, 90.0, 111.0}};
double notesD[3][3] = {{0.0, 0.0, 0.0}, {0.0, 1.0, 2.0}, {0.0, 2.0, 4.0}};
double notesX[3] = {0.0, 1.0, 2.0};

int main (int argc, char *argv[])
{
    int row_idx, col_jdx, n=DEFAULT_DIM; // indexes into arrays and dimention of linear system
    int count=0;

    // Timing declarations
    struct timespec start, end;
    double fstart=0.0, fend=0.0;


    if(argc > 1)
    {
       printf("Creating custom dimension matrix and vector: argc=%d, argv[0]=%s, argv[1]=%s, argv[2]=%s\n",
              argc, argv[0], argv[1], argv[2]);
       n=atoi(argv[1]); count=0;

       if(argc == 3) thread_count=atoi(argv[2]); // user selected number of threads

       if(n == 3)
       {
           // for 3x3, use example from notes and turn on the debug!
           debug=1;

           printf("Using %d x %d example from CSCI 551 notes\n", n, n);
           for(row_idx = 0; row_idx < n; row_idx++)
           {
                for(col_jdx = 0; col_jdx < n; col_jdx++)
                {
                    default_mat[row_idx][col_jdx] = notesA[row_idx][col_jdx];
                    alt_mat[row_idx][col_jdx] = notesD[row_idx][col_jdx];
                }
                default_vect[row_idx] = notesX[row_idx];
           }
       }
       else
       {
           // for sizes other than 3x3, generate pattern
           printf("Generating %d x %d example vector and matrix\n", n, n);

           for(row_idx = 0; row_idx < n; row_idx++)
           {
                for(col_jdx = 0; col_jdx < n; col_jdx++)
                {
                    default_mat[row_idx][col_jdx] = count++;
                    alt_mat[row_idx][col_jdx] = count++;
                }
                default_vect[row_idx] = count;
           }
       }
    }
    else
    {
        printf("Using DEFAULT %d x %d example vector and matrix\n", DEFAULT_DIM, DEFAULT_DIM);
        count=0;

        for(row_idx = 0; row_idx < DEFAULT_DIM; row_idx++)
        {
            for(col_jdx = 0; col_jdx < DEFAULT_DIM; col_jdx++)
            {
                default_mat[row_idx][col_jdx] = count++;
            }
            default_vect[row_idx] = count;
        }
    }


    // TIME vmult: Pass in thread_count = 1 for sequential testing
    //
    if(debug)
    {
        printf("\nCoefficient Matrix A:\n");
        matrix_print(n, n, alt_mat);
        printf("\nVector x:\n");
        vector_print(n, default_vect);
        printf("\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &start);

    // ******* Call vmult here ***********************
    //
	// this function does not speed-up well with OpenMP without modification to divide up work
//#pragma omp parallel num_threads(thread_count)
    vmult(n, alt_mat, default_vect);

    clock_gettime(CLOCK_MONOTONIC, &end);
    fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
    fend=end.tv_sec + (end.tv_nsec / 1000000000.0);
    printf("vmult %d took %lf seconds, with %d workers\n", n, (fend-fstart), thread_count);



    // TIME mmult: Pass in thread_count = 1 for sequential testing
    if(debug)
    {
        printf("\nMatrix A:\n");
        matrix_print(n, n, default_mat);
        printf("\nMatrix B:\n");
        matrix_print(n, n, default_mat);
        printf("\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &start);

    // ******* Call mmult here ***********************
    //
	// this function does not speed-up well with OpenMP without modification to divide up work
//#pragma omp parallel num_threads(thread_count)
    mmult(n, default_mat, default_mat);

    clock_gettime(CLOCK_MONOTONIC, &end);
    fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
    fend=end.tv_sec + (end.tv_nsec / 1000000000.0);
    printf("mmult %d took %lf seconds, with %d workers\n", n, (fend-fstart), thread_count);

    return(0);
} 


void vmult(int n, double A[][MAX_DIM], double x[MAX_DIM])
{
    int row_idx, col_jdx;
    double rhs[n], temp;

    // for all rows - this loop speeds up well with OpenMP
#pragma omp parallel for num_threads(thread_count) private(row_idx, col_jdx) shared(n)
    for (row_idx=0; row_idx < n; ++row_idx)
    {
        rhs[row_idx] = 0.0; temp=0.0;

        // sum up row's column coefficient x solution vector element
        // as we would do for any matrix * vector operation which yields a vector,
        // which should be the RHS
        for (col_jdx=0; col_jdx < n; ++col_jdx)
        {
            temp += A[row_idx][col_jdx] * x[col_jdx];
        }
        rhs[row_idx]=temp;
    }

    if(debug)
    {
        // Computed RHS
        printf("\nComputed RHS is:\n");
        vector_print(n, rhs);
    }
}


void mmult(int n, double A[][MAX_DIM], double B[][MAX_DIM])
{
    int row_idx, col_jdx, coeff_idx;
    double temp;

    // for all rows - this loop speeds-up well with OpenMP
#pragma omp parallel for num_threads(thread_count) private(row_idx, col_jdx, coeff_idx) shared(n)
    for (row_idx=0; row_idx < n; ++row_idx)
    {
        for (col_jdx=0; col_jdx < n; ++col_jdx)
        {
            if(debug) printf("C[%d][%d]:\n", row_idx, col_jdx);

            for(coeff_idx=0; coeff_idx < n; ++coeff_idx)
            {
                C[row_idx][col_jdx] += A[row_idx][coeff_idx] * B[coeff_idx][col_jdx];
                if(debug) printf("A[%d][%d]=%lf, b[%d][%d]=%lf\n", row_idx, coeff_idx, A[row_idx][coeff_idx], coeff_idx, col_jdx, B[coeff_idx][col_jdx]);
            }
        }
    }

    if(debug)
    {
        printf("\nComputed C is:\n");
        printf("A=\n"); matrix_print(n, n, A); printf("\n");
        printf("B=\n"); matrix_print(n, n, B); printf("\n");
        printf("C=\n"); matrix_print(n, n, C); printf("\n");
    }

}


void  matrix_print (int nr, int nc, double A[][MAX_DIM])
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



void  vector_print (int nr, double x[MAX_DIM])
{
    int row_idx;
  
    for (row_idx = 0; row_idx < nr; row_idx++) 
    {
    	printf ("%9.4f  \n", x[row_idx]);
    }

    printf("\n");  // Insert a new line at the end
}
