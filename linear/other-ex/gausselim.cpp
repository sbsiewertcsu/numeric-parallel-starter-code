/*
*   GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING ALGORITHM 6.2
*
*   https://www3.nd.edu/~zxu2/acms40390hw/Gaussian_pivot.cpp
*
*   To solve the n by n linear system
*
*   E1:  A[1,1] X[1] + A[1,2] X[2] +...+ A[1,n] X[n] = A[1,n+1]
*   E2:  A[2,1] X[1] + A[2,2] X[2] +...+ A[2,n] X[n] = A[2,n+1]
*   :
*   .
*   EN:  A[n,1] X[1] + A[n,2] X[2] +...+ A[n,n] X[n] = A[n,n+1]
*
*   INPUT:   number of unknowns and equations n; augmented
*            matrix A = (A(I,J)) where 1<=I<=n and 1<=J<=n+1.
*
*   OUTPUT:  solution x(1), x(2),...,x(n) or a message that the
*            linear system has no unique solution.
*   EXample 6.2.1 is solved here.
*/

#include<stdio.h>
#include<math.h>
#include<iostream> 

#define ZERO 1.0E-20
#define true 1
#define false 0

#define N   2   /* Number of equations */

static double absval(double);
static void   OUTPUT(int, int, int *, double *, double [][N+1]);

using namespace std;

int main()
{
   double A[N][N+1] = { {0.003, 59.14, 59.17},
                        {5.291, -6.13, 46.78}
                         };
   double  X[N];

   double AMAX,XM,SUM;  /* AMAX save  a_{pk} = max_{k<=i<=n} |a_{ik}^{(k)}| */
   int NROW[N]; /* save  order of rows due to pivoting */

   int M,ICHG,I,NN,IMAX,J,JJ,IP,JP,NCOPY,I1,J1,N1,K,N2,LL,KK,OK = true;

      M = N + 1;
      /* STEP 1 */
      for (I=1; I<=N; I++) NROW[I-1] = I;

      //Add your code to compute scale factor for each row here

      /* initialize row pointer */
      NN = N - 1;
      ICHG = 0;
      I = 1; 
      /* STEP 2 */
      while ((OK) && (I <= NN)) {
         /* STEP 3 */
         // For the scaled partial pivoting, AMAX should be scaled by the scale factor  

         IMAX = NROW[I-1];
         AMAX = absval(A[IMAX-1][I-1]);
         IMAX = I;
         JJ = I + 1;
         for (IP=JJ; IP<=N; IP++) 
         {
            JP = NROW[IP-1];
            if (absval(A[JP-1][I-1]) > AMAX) 
            {
               AMAX = absval(A[JP-1][I-1]);
               IMAX = IP;
            }
         
         }  
         /* STEP 4 */
         if (AMAX <= ZERO) OK = false;
         else { 
            /* STEP 5 */
            /* simulate row interchange */
            if ( NROW[I-1] != NROW[IMAX-1]) 
            {
               ICHG = ICHG + 1;
               NCOPY = NROW[I-1];
               NROW[I-1] = NROW[IMAX-1];
               NROW[IMAX-1] = NCOPY;
            }
            I1 = NROW[I-1];
            /* STEP 6 --- Gaussian elimination step */
            for (J=JJ; J<=N; J++) 
            {
               J1 = NROW[J-1];
               /* STEP  7 */
               XM = A[J1-1][I-1] / A[I1-1][I-1];
               /* STEP 8 */
               for (K=JJ; K<=M; K++) 
                  A[J1-1][K-1] = A[J1-1][K-1] - XM * A[I1-1][K-1];
               /* Multiplier XM could be saved in A[J1-1,I-1]  */
               A[J1-1][I-1] = 0.0;
            }  
         }  
         I++;
      }


      if (OK) 
      {
         /* STEP 9 */
         N1 = NROW[N-1];
         if (absval(A[N1-1][N-1]) <= ZERO) OK = false;
         /* system has no unique solution */
         else 
         {
            /* STEP 10 */
            /* start backward substitution */
            X[N-1] = A[N1-1][M-1] / A[N1-1][N-1];
            /* STEP 11 */
            for (K=1; K<=NN; K++) {
               I = NN - K + 1;
               JJ = I + 1;
               N2 = NROW[I-1];
               SUM = 0.0;
               for (KK=JJ; KK<=N; KK++) {
                  SUM = SUM - A[N2-1][KK-1] * X[KK-1];
               }  
               X[I-1] = (A[N2-1][N] + SUM) / A[N2-1][I-1];
            }  
            /* STEP 12 */
            /* procedure completed successfully */
            OUTPUT(M, ICHG, NROW, X, A);
         }  
      }
      if (!OK) cout<<"System has no unique solution"<<endl;

   return 0;
}

/* Absolute Value Function */
static double absval(double val)
{
   if (val >= 0) return val;
   else return -val;
}

static void OUTPUT(int M, int ICHG, int *NROW, double *X, double A[][N+1])
{
   int I, J, FLAG;

   cout<<"GAUSSIAN ELIMINATION - PARTIAL PIVOTING"<<endl<<endl;
   cout<<"The reduced system - output by rows:"<<endl;

   for (I=1; I<=N; I++) {
      for (J=1; J<=N; J++) printf(" %11.8f", A[I-1][J-1]);
      cout<<endl;
   }

   cout<<endl<<endl<<"Has solution vector:"<<endl;
   for (I=1; I<=N; I++) 
   {
      printf("  %12.8f", X[I-1]);
   }

   printf("\nwith %d row interchange(s)\n", ICHG);
   printf("\nThe rows have been logically re-ordered to:\n");
   for (I=1; I<=N; I++) 
       printf(" %2d", NROW[I-1]); 
   cout<<endl;
}


