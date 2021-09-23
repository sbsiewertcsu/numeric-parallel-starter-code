#include <stdio.h>

#define DIMX (8)
#define DIMY (6)

#define max(X, Y) ((X) > (Y) ? (X) : (Y))

unsigned int P[max(DIMX, DIMY)][max(DIMX, DIMY)];
unsigned int TP[max(DIMX, DIMY)][max(DIMX, DIMY)];
unsigned int RRP[max(DIMX, DIMY)][max(DIMX, DIMY)];
unsigned int RLP[max(DIMX, DIMY)][max(DIMX, DIMY)];

void zeroIntMat(unsigned int Mat[][max(DIMX, DIMY)]);
void fillIntMat(unsigned int Mat[][max(DIMX, DIMY)]);
void printIntMat(unsigned int Mat[][max(DIMX, DIMY)]);
void transposeIntMat(unsigned int Mat[][max(DIMX, DIMY)], unsigned int TMat[][max(DIMX, DIMY)]);
void swapColIntMat(unsigned int Mat[][max(DIMX, DIMY)], unsigned int TMat[][max(DIMX, DIMY)]);
void swapRowIntMat(unsigned int Mat[][max(DIMX, DIMY)], unsigned int TMat[][max(DIMX, DIMY)]);


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


void swapRowIntMat(unsigned int Mat[][max(DIMX, DIMY)], unsigned int TMat[][max(DIMX, DIMY)])
{
    int idx, jdx;

    for(idx=0; idx<max(DIMX, DIMY); idx++)       
        for(jdx=0; jdx<max(DIMX, DIMY); jdx++)  
        {
            // copy into TMat and swap row values         
            TMat[idx][jdx]=Mat[max(DIMX, DIMY)-1-idx][jdx];
        }
}


void swapColIntMat(unsigned int Mat[][max(DIMX, DIMY)], unsigned int TMat[][max(DIMX, DIMY)])
{
    int idx, jdx;

    for(idx=0; idx<max(DIMX, DIMY); idx++)       
        for(jdx=0; jdx<max(DIMX, DIMY); jdx++)  
        {
            // copy into TMat and swap column values         
            TMat[idx][jdx]=Mat[idx][max(DIMX, DIMY)-1-jdx];
        }
}


void fillIntMat(unsigned int Mat[][max(DIMX, DIMY)])
{
    int cnt, idx, jdx;

    for(idx=0; idx<max(DIMX, DIMY); idx++)       
        for(jdx=0; jdx<max(DIMX, DIMY); jdx++)
        {
            Mat[idx][jdx]=cnt;
            cnt++;
        }
}


void zeroIntMat(unsigned int Mat[][max(DIMX, DIMY)])
{
    int idx, jdx;

    for(idx=0; idx<max(DIMX, DIMY); idx++)       
        for(jdx=0; jdx<max(DIMX, DIMY); jdx++)
        {
            Mat[idx][jdx]=0;
        }
}


void transposeIntMat(unsigned int Mat[][max(DIMX, DIMY)], unsigned int TMat[][max(DIMX, DIMY)])
{
    int idx, jdx;

    for(idx=0; idx<max(DIMX, DIMY); idx++)
        for(jdx=0; jdx<max(DIMX, DIMY); jdx++)
        {
            // transpose row as column
            TMat[jdx][idx]=Mat[idx][jdx];
        }
}


void printIntMat(unsigned int Mat[][max(DIMX, DIMY)])
{
    int idx, jdx;

    for(idx=0; idx<max(DIMX, DIMY); idx++)
    {
         printf("\n");
         for(jdx=0; jdx<max(DIMX, DIMY); jdx++)
             printf("%02d ", Mat[idx][jdx]);
    }
    printf("\n\n");;
}

