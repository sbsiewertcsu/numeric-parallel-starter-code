#include <stdio.h>

#define DIM (3)

unsigned int X[DIM][DIM];
unsigned int Y[DIM][DIM];

void printIntMat(unsigned int Mat[][DIM]);


int main(void)
{
   int idx, jdx;
   unsigned int cnt=0;

    for(idx=0; idx<DIM; idx++)       
        for(jdx=0; jdx<DIM; jdx++)
        {
            X[idx][jdx]=cnt;
            Y[idx][jdx]=0;
            cnt++;
        }


    for(idx=0; idx<DIM; idx++)       
        for(jdx=0; jdx<DIM; jdx++)  
        {
            // transpose row as column                      
            Y[jdx][idx]=X[idx][jdx]; 
            
            // transpose column as row                
            //Y[idx][jdx]=X[jdx][idx]; 
        }

    printIntMat(X);
    printIntMat(Y);

    return 0;
}


void printIntMat(unsigned int Mat[][DIM])
{
    int idx, jdx;

    printf("\n");

    for(idx=0; idx<DIM; idx++)
    {
         printf("\n");       
         for(jdx=0; jdx<DIM; jdx++)
             printf("%d", Mat[idx][jdx]);
    }
    printf("\n");;
}


