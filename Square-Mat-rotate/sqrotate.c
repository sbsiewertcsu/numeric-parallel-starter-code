#include <stdio.h>

// Square dimension
#define DIM (10)

unsigned int P[DIM][DIM];
unsigned int TP[DIM][DIM];
unsigned int RRP[DIM][DIM];
unsigned int RLP[DIM][DIM];

void zeroIntMat(unsigned int Mat[][DIM]);
void fillIntMat(unsigned int Mat[][DIM]);
void printIntMat(unsigned int Mat[][DIM]);
void transposeIntMat(unsigned int Mat[][DIM], unsigned int TMat[][DIM]);
void swapColIntMat(unsigned int Mat[][DIM], unsigned int TMat[][DIM]);
void swapRowIntMat(unsigned int Mat[][DIM], unsigned int TMat[][DIM]);


void main(void)
{
    fillIntMat(P);
    printf("Matrix P=");printIntMat(P);

    transposeIntMat(P, TP);
    //printf("Transpose (rotation about left-to-right diagonal) of P=");printIntMat(TP);
    printf("Transpose of P=");printIntMat(TP);

    swapColIntMat(TP, RRP);
    swapRowIntMat(TP, RLP);

    printf("P=");printIntMat(P);
    //printf("Rotate Right (column rotate after TP), P=");printIntMat(RRP);
    printf("Rotate Right P=");printIntMat(RRP);
    //printf("Rotate Left (rown rotate after TP), P=");printIntMat(RLP);
    printf("Rotate Left P=");printIntMat(RLP);
}


void swapRowIntMat(unsigned int Mat[][DIM], unsigned int TMat[][DIM])
{
    int idx, jdx;

    for(idx=0; idx<DIM; idx++)       
        for(jdx=0; jdx<DIM; jdx++)  
        {
            // copy into TMat and swap row values         
            TMat[idx][jdx]=Mat[DIM-1-idx][jdx];
        }
}


void swapColIntMat(unsigned int Mat[][DIM], unsigned int TMat[][DIM])
{
    int idx, jdx;

    for(idx=0; idx<DIM; idx++)       
        for(jdx=0; jdx<DIM; jdx++)  
        {
            // copy into TMat and swap column values         
            TMat[idx][jdx]=Mat[idx][DIM-1-jdx];
        }
}


void fillIntMat(unsigned int Mat[][DIM])
{
    int cnt, idx, jdx;

    for(idx=0; idx<DIM; idx++)       
        for(jdx=0; jdx<DIM; jdx++)
        {
            Mat[idx][jdx]=cnt;
            cnt++;
        }
}


void zeroIntMat(unsigned int Mat[][DIM])
{
    int idx, jdx;

    for(idx=0; idx<DIM; idx++)       
        for(jdx=0; jdx<DIM; jdx++)
        {
            Mat[idx][jdx]=0;
        }
}


void transposeIntMat(unsigned int Mat[][DIM], unsigned int TMat[][DIM])
{
    int idx, jdx;

    for(idx=0; idx<DIM; idx++)
        for(jdx=0; jdx<DIM; jdx++)
        {
            // transpose row as column
            TMat[jdx][idx]=Mat[idx][jdx];
        }
}


void printIntMat(unsigned int Mat[][DIM])
{
    int idx, jdx;

    for(idx=0; idx<DIM; idx++)
    {
         printf("\n");
         for(jdx=0; jdx<DIM; jdx++)
             printf("%02d ", Mat[idx][jdx]);
    }
    printf("\n\n");;
}

