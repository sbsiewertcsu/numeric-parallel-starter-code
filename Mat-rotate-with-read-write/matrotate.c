#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

// This is the dimension of a playing card with a 3:4 Aspect Ratio (standing upright)
#define DIMX (690)
#define DIMY (920)

#define max(X, Y) ((X) > (Y) ? (X) : (Y))
#define SQDIM (max(DIMX, DIMY))

unsigned char P[SQDIM][SQDIM];     // Pixel array of gray values
unsigned char TP[SQDIM][SQDIM];    // Transpose of Pixel array
unsigned char RRP[SQDIM][SQDIM];   // Rotation Right of Pixel array
unsigned char RLP[SQDIM][SQDIM];   // Rotation Left of Pixel array

void zeroPixMat(unsigned char Mat[][SQDIM]);
void fillPixMat(unsigned char Mat[][SQDIM]);
void printPixMat(unsigned char Mat[][SQDIM], int square_size);
void transposePixMat(unsigned char Mat[][SQDIM], unsigned char TMat[][SQDIM]);
void swapColPixMat(unsigned char Mat[][SQDIM], unsigned char TMat[][SQDIM], int square_size);
void swapRowPixMat(unsigned char Mat[][SQDIM], unsigned char TMat[][SQDIM], int square_size);

// Simple demonstration function for small array rotation
void demoRotate(int square_size);

// PGM file utilities with simple byte by byte I/O
void readPGMHeaderSimple(int fdin, char *header);
void printPGMHeader(char *header);
void readPGMDataSimple(int fdin, unsigned char Mat[][SQDIM]);
void writePGMSimple(int fdout, char *header, unsigned char Mat[][SQDIM]);

// PGM file utilities with fast large read and write I/O
void readPGMHeaderFast(int fdin, char *header);
void readPGMDataFast(int fdin, unsigned char Mat[][SQDIM]);
void writePGMFast(int fdout, char *header, unsigned char Mat[][SQDIM]);
void writePGMFastSquare(int fdout, char *header, unsigned char Mat[][SQDIM]);


int main(int argc, char *argv[])
{
    int fdin, fdout, rowIdx, colIdx;
    //int bytesRead, bytesLeft, bytesWritten;
    char header[80];

    // initialize Pixel array with all zeros
    zeroPixMat(P);

    if(argc < 3) {
        printf("Use: matrotate <inputfile> <outputfile>\n");
        exit(-1);
    }

    else {
         // open binary file to read in data instead of using test pattern data
         if((fdin = open(argv[1], O_RDONLY, 0644)) < 0) {
             printf("Error opening %s\n", argv[1]); exit(-1);
         }

         // open binary file to write out data
         if((fdout = open(argv[2], O_WRONLY | O_CREAT, 0644)) < 0) {
             printf("Error opening %s\n", argv[2]); exit(-1);
         }
    }

    // Determine playing card type based upon filename.  Using filename convention is the most
    // simple and reasonable approach, otherwise we could use color and markings as indicators
    // with PPM, but this is not necessary and is more of an image processing challenge.
    //     Value: A=Ace, 2-10, J=Jack, Q=Queen, K=King
    //     Suit:  D=Diamonds, H=Hearts, S=Spades, C=Clubs
    //     if the PGM is for a club or spade (black and white), note that it must be rotated RIGHT
    //     if the PGM is for a heart or diamond (red and white),  note that it must be rotated LEFT


    // read in the PGM data here
    readPGMHeaderSimple(fdin, header);
    //readPGMDataSimple(fdin, P);
    readPGMDataFast(fdin, P);
    close(fdin);

    // Update header to be square 920x920
    header[26]='9'; header[27]='2'; header[28]='0';
    printf("\nUpdated SQUARE HEADER:\n");
    printPGMHeader(header);

    // write out modified PGM data here
    //writePGMSimple(fdout, header, P);
    //writePGMFast(fdout, header, P);
    writePGMFastSquare(fdout, header, P);
    close(fdout);

    printf("Read and then write of unmodified PGM done\n");

    // demonstrate rotation with a smaller verification size
    demoRotate(8);

    printf("Demonstration done\n");
}


void swapRowPixMat(unsigned char Mat[][SQDIM], unsigned char TMat[][SQDIM], int square_size)
{
    int idx, jdx;

    for(idx=0; idx<square_size; idx++)       
        for(jdx=0; jdx<square_size; jdx++)  
        {
            // copy into TMat and swap row values         
            TMat[idx][jdx]=Mat[square_size-1-idx][jdx];
        }
}


void swapColPixMat(unsigned char Mat[][SQDIM], unsigned char TMat[][SQDIM], int square_size)
{
    int idx, jdx;

    for(idx=0; idx<square_size; idx++)       
        for(jdx=0; jdx<square_size; jdx++)  
        {
            // copy into TMat and swap column values         
            TMat[idx][jdx]=Mat[idx][square_size-1-jdx];
        }
}


void fillPixMat(unsigned char Mat[][SQDIM])
{
    int cnt=0, idx, jdx;

    for(idx=0; idx<SQDIM; idx++)       
        for(jdx=0; jdx<SQDIM; jdx++)
        {
            Mat[idx][jdx]=cnt;
            cnt++;
        }
}


void zeroPixMat(unsigned char Mat[][SQDIM])
{
    int idx, jdx;

    for(idx=0; idx<SQDIM; idx++)       
        for(jdx=0; jdx<SQDIM; jdx++)
        {
            Mat[idx][jdx]=0;
        }
}


void transposePixMat(unsigned char Mat[][SQDIM], unsigned char TMat[][SQDIM])
{
    int idx, jdx;

    for(idx=0; idx<SQDIM; idx++)
        for(jdx=0; jdx<SQDIM; jdx++)
        {
            // transpose row as column
            TMat[jdx][idx]=Mat[idx][jdx];
        }
}


void printPixMat(unsigned char Mat[][SQDIM], int square_size)
{
    int idx, jdx;

    for(idx=0; idx<square_size; idx++)
    {
         printf("\n");
         for(jdx=0; jdx<square_size; jdx++)
             printf("%03d ", Mat[idx][jdx]);
    }
    printf("\n\n");;
}

void readPGMHeaderSimple(int fdin, char *header)
{
    int bytesRead, bytesLeft, bytesWritten;

    printf("Reading PGM header here\n");

    // header on each card is 38 bytes
    bytesLeft=38;
    bytesRead=read(fdin, (void *)header, bytesLeft);

    if(bytesRead < bytesLeft)
        exit(-1);
    else
        printf("header=%s\n", header);

}

void printPGMHeader(char *header)
{
    printf("%s", header);
}

void readPGMDataSimple(int fdin, unsigned char Mat[][SQDIM])
{
    int bytesRead, bytesLeft, bytesWritten;
    int rowIdx, colIdx;

    printf("Reading PGM data here\n");

    // now read in all of the data
    bytesRead=0;

    // read in the actual size and aspect ratio of original card
    for(rowIdx = 0; rowIdx < DIMY; rowIdx++)
    {
        for(colIdx = 0; colIdx < DIMX; colIdx++)
            bytesRead=read(fdin, (void *)&P[rowIdx][colIdx], 1);
            bytesRead++;
    }

}


void readPGMDataFast(int fdin, unsigned char Mat[][SQDIM])
{
    int bytesRead, bytesLeft, bytesWritten;
    int rowIdx, colIdx;

    printf("Reading PGM data here\n");

    // now read in all of the data
    bytesRead=0;

    // read in whole rows at a time to speed up
    for(rowIdx = 0; rowIdx < DIMY; rowIdx++)
    {
        bytesRead=read(fdin, (void *)&P[rowIdx][0], DIMX);
    }

}


void writePGMSimple(int fdout, char *header, unsigned char Mat[][SQDIM])
{
    int bytesRead, bytesLeft, bytesWritten;
    int rowIdx, colIdx;

    printf("Would write out a header and data here\n");
    bytesLeft=38;

    bytesWritten=write(fdout, (void *)header, bytesLeft);
    
    printf("wrote %d bytes for header\n", bytesWritten);

    // now write out all of the data
    bytesWritten=0;

    for(rowIdx = 0; rowIdx < DIMY; rowIdx++)
    {
        for(colIdx = 0; colIdx < DIMX; colIdx++)
            bytesWritten=write(fdout, (void *)&P[rowIdx][colIdx], 1);
            bytesWritten++;
    }
}

void writePGMFast(int fdout, char *header, unsigned char Mat[][SQDIM])
{
    int bytesRead, bytesLeft, bytesWritten;
    int rowIdx, colIdx;

    printf("Would write out a header and data here\n");
    bytesLeft=38;

    bytesWritten=write(fdout, (void *)header, bytesLeft);
    
    printf("wrote %d bytes for header\n", bytesWritten);

    // now write out all of the data
    bytesWritten=0;

    for(rowIdx = 0; rowIdx < DIMY; rowIdx++)
    {
        bytesWritten=write(fdout, (void *)&P[rowIdx][0], DIMX);
    }
}

void writePGMFastSquare(int fdout, char *header, unsigned char Mat[][SQDIM])
{
    int bytesRead, bytesLeft, bytesWritten;
    int rowIdx, colIdx;

    printf("Would write out a header and data here\n");
    bytesLeft=38;

    bytesWritten=write(fdout, (void *)header, bytesLeft);
    
    printf("wrote %d bytes for header\n", bytesWritten);

    // now write out all of the data
    bytesWritten=0;

    for(rowIdx = 0; rowIdx < SQDIM; rowIdx++)
    {
        bytesWritten=write(fdout, (void *)&P[rowIdx][0], SQDIM);
        if(bytesWritten < SQDIM)
        {
            printf("ERROR in write\n"); exit(-1);
        }
    }
}


void demoRotate(int square_size)
{
    // initialize global P array with test data for maximum dimensions
    fillPixMat(P);

    // print small test area for verification
    printf("Matrix P=");printPixMat(P, square_size);

    // TRANSPOSE entire maximum dimension square array
    transposePixMat(P, TP);
    //printf("Transpose (rotation about left-to-right diagonal) of P=");printPixMat(TP);
    printf("Transpose of P=");printPixMat(TP, square_size);

    // PIVOT entire maximum dimension square array
    swapColPixMat(TP, RRP, square_size);
    swapRowPixMat(TP, RLP, square_size);

    // print small test area for verification
    printf("P=");printPixMat(P, square_size);
    //printf("Rotate Right (column rotate after TP), P=");printPixMat(RRP, square_size);
    printf("Rotate Right P=");printPixMat(RRP, square_size);
    //printf("Rotate Left (rown rotate after TP), P=");printPixMat(RLP, square_size);
    printf("Rotate Left P=");printPixMat(RLP, square_size);
}


