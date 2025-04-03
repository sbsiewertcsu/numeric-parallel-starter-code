// https://www.codesansar.com/numerical-methods/gauss-jordan-method-using-c-programming.htm
//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define SIZE 10

int main(void)
{
    float a[SIZE][SIZE], x[SIZE], ratio;
    int i,j,k,n;
    int thread_count=1;

   
    /* Inputs */

    /* 0. Reading number of threads */
    printf("Enter number of OMP threads: ");
    scanf("%d", &thread_count);
    printf("%d\n", thread_count);

    /* 1. Reading number of unknowns */
    printf("Enter number of unknowns: ");
    scanf("%d", &n);
    printf("%d\n", n);

    /* 2. Reading Augmented Matrix */
    printf("Enter coefficients of Augmented Matrix:\n");
    for(i=1;i<=n;i++)
    {
        for(j=1;j<=n+1;j++)
        {
            printf("a[%d][%d] = ",i,j);
            scanf("%f", &a[i][j]);
            printf("%020.15lf\n", a[i][j]);

            if((i == j) && a[i][i] == 0.0)
            {
                printf("Mathematical Error!");
                exit(0);
            }
        }
    }

    /* 3. Applying Gauss Jordan Elimination */
#pragma omp parallel for num_threads(thread_count) private(i, j, ratio) shared(n)
    for(i=1;i<=n;i++)
    {
        // Do check for ZERO in diagonal on input to make loop simpler to 
        // make parallel
        //if(a[i][i] == 0.0)
        //{
        //    printf("Mathematical Error!");
        //    exit(0);
        //}

        for(j=1;j<=n;j++)
        {
            if(i!=j)
            {
                ratio = a[j][i]/a[i][i];
                for(k=1;k<=n+1;k++)
                {
                    a[j][k] = a[j][k] - ratio*a[i][k];
                }
            }
        }
    }

    /* 4. Obtaining Solution */
#pragma omp parallel for num_threads(thread_count) private(i) shared(n)
    for(i=1;i<=n;i++)
    {
        x[i] = a[i][n+1]/a[i][i];
    }

    /* Displaying Solution */
    printf("\nSolution:\n");
    for(i=1;i<=n;i++)
    {
        printf("x[%d] = %0.3f\n",i, x[i]);
    }

    /* Add verification of Solution here */

    getchar();
    return(0);
}

