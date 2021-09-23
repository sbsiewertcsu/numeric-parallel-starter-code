// Sam Siewert, July 16, 2020
//
// Based on basic PSF convolution as documented in DSP Engineer's Handbook
//
// http://www.dspguide.com/pdfbook.htm
//
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>


//#define IMG_HEIGHT (240)
//#define IMG_WIDTH (320)

#define IMG_HEIGHT (960)
#define IMG_WIDTH (1280)

#define ITERATIONS (90)

#define FAST_IO

typedef double FLOAT;

typedef unsigned int UINT32;
typedef unsigned long long int UINT64;
typedef unsigned char UINT8;

// PPM Edge Enhancement Code
//
UINT8 header[22];
UINT8 R[IMG_HEIGHT*IMG_WIDTH];
UINT8 G[IMG_HEIGHT*IMG_WIDTH];
UINT8 B[IMG_HEIGHT*IMG_WIDTH];
UINT8 convR[IMG_HEIGHT*IMG_WIDTH];
UINT8 convG[IMG_HEIGHT*IMG_WIDTH];
UINT8 convB[IMG_HEIGHT*IMG_WIDTH];

// PPM image array with channels
UINT8 RGB[IMG_HEIGHT*IMG_WIDTH*3];

// controls sharpness
// increase from K=4.0 and F=8.0 for sharper edges
#define K 4.0
#define F 8.0
//#define F 80.0

FLOAT PSF[9] = {-K/F, -K/F, -K/F, -K/F, K+1.0, -K/F, -K/F, -K/F, -K/F};


int main(int argc, char *argv[])
{
    int fdin, fdout, bytesRead=0, bytesWritten=0, bytesLeft, i, j, iter, rc, pixel, readcnt=0, writecnt=0;
    UINT64 microsecs=0, millisecs=0;
    FLOAT temp, fstart, fnow;
    struct timespec start, now;
    int thread_count=4;

    clock_gettime(CLOCK_MONOTONIC, &start);
    fstart = (FLOAT)start.tv_sec  + (FLOAT)start.tv_nsec / 1000000000.0;
    
    if(argc < 3)
    {
       printf("Usage: sharpen input_file.ppm output_file.ppm\n");
       exit(-1);
    }
    else
    {
        if((fdin = open(argv[1], O_RDONLY, 0644)) < 0)
        {
            printf("Error opening %s\n", argv[1]);
        }
        //else
        //    printf("File opened successfully\n");

        if((fdout = open(argv[2], (O_RDWR | O_CREAT), 0666)) < 0)
        {
            printf("Error opening %s\n", argv[1]);
        }
        //else
        //    printf("Output file=%s opened successfully\n", "sharpen.ppm");
    }

    bytesLeft=21;

    //printf("Reading header\n");

    // read in all data
    do
    {
        //printf("bytesRead=%d, bytesLeft=%d\n", bytesRead, bytesLeft);
        bytesRead=read(fdin, (void *)header, bytesLeft);
        bytesLeft -= bytesRead;
    } while(bytesLeft > 0);

    header[21]='\0';

    printf("header = %s\n", header); 


#ifdef FAST_IO

    bytesRead=0;
    bytesLeft=IMG_HEIGHT*IMG_WIDTH*3;
    readcnt=0;
    printf("START: read %d, bytesRead=%d, bytesLeft=%d\n", readcnt, bytesRead, bytesLeft);

    // Read in RGB data in large chunks, requesting all and reading residual
    do
    {
        bytesRead=read(fdin, (void *)&RGB[bytesRead], bytesLeft);
        bytesLeft -= bytesRead;
        readcnt++;

        printf("read %d, bytesRead=%d, bytesLeft=%d\n", readcnt, bytesRead, bytesLeft);

    } while((bytesLeft > 0) && (readcnt < 3));

    printf("END: read %d, bytesRead=%d, bytesLeft=%d\n", readcnt, bytesRead, bytesLeft);

    // create in memory copy from input by channel
    for(i=0, pixel=0; i<IMG_HEIGHT*IMG_WIDTH; i++, pixel+=3)
    {
        R[i]=RGB[pixel+0]; convR[i]=R[i];
        G[i]=RGB[pixel+1]; convG[i]=G[i];
        B[i]=RGB[pixel+2]; convB[i]=B[i];
    }

#else

    // Read RGB data - Very slow one byte at time!
    for(i=0; i<IMG_HEIGHT*IMG_WIDTH; i++)
    {
        rc=read(fdin, (void *)&R[i], 1); convR[i]=R[i];
        rc=read(fdin, (void *)&G[i], 1); convG[i]=G[i];
        rc=read(fdin, (void *)&B[i], 1); convB[i]=B[i];
    }
#endif


    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec  + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("\nstart test at %lf\n", fnow-fstart);
    clock_gettime(CLOCK_MONOTONIC, &start);
    fstart = (FLOAT)start.tv_sec  + (FLOAT)start.tv_nsec / 1000000000.0;

#pragma omp parallel for num_threads(thread_count)
    for(iter=0; iter < ITERATIONS; iter++)
    {
        // Skip first and last row, no neighbors to convolve with
        for(i=1; i<((IMG_HEIGHT)-1); i++)
        {

            // Skip first and last column, no neighbors to convolve with
            for(j=1; j<((IMG_WIDTH)-1); j++)
            {
                temp=0;
                temp += (PSF[0] * (FLOAT)R[((i-1)*IMG_WIDTH)+j-1]);
                temp += (PSF[1] * (FLOAT)R[((i-1)*IMG_WIDTH)+j]);
                temp += (PSF[2] * (FLOAT)R[((i-1)*IMG_WIDTH)+j+1]);
                temp += (PSF[3] * (FLOAT)R[((i)*IMG_WIDTH)+j-1]);
                temp += (PSF[4] * (FLOAT)R[((i)*IMG_WIDTH)+j]);
                temp += (PSF[5] * (FLOAT)R[((i)*IMG_WIDTH)+j+1]);
                temp += (PSF[6] * (FLOAT)R[((i+1)*IMG_WIDTH)+j-1]);
                temp += (PSF[7] * (FLOAT)R[((i+1)*IMG_WIDTH)+j]);
                temp += (PSF[8] * (FLOAT)R[((i+1)*IMG_WIDTH)+j+1]);
	        if(temp<0.0) temp=0.0;
	        if(temp>255.0) temp=255.0;
	        convR[(i*IMG_WIDTH)+j]=(UINT8)temp;

                temp=0;
                temp += (PSF[0] * (FLOAT)G[((i-1)*IMG_WIDTH)+j-1]);
                temp += (PSF[1] * (FLOAT)G[((i-1)*IMG_WIDTH)+j]);
                temp += (PSF[2] * (FLOAT)G[((i-1)*IMG_WIDTH)+j+1]);
                temp += (PSF[3] * (FLOAT)G[((i)*IMG_WIDTH)+j-1]);
                temp += (PSF[4] * (FLOAT)G[((i)*IMG_WIDTH)+j]);
                temp += (PSF[5] * (FLOAT)G[((i)*IMG_WIDTH)+j+1]);
                temp += (PSF[6] * (FLOAT)G[((i+1)*IMG_WIDTH)+j-1]);
                temp += (PSF[7] * (FLOAT)G[((i+1)*IMG_WIDTH)+j]);
                temp += (PSF[8] * (FLOAT)G[((i+1)*IMG_WIDTH)+j+1]);
	        if(temp<0.0) temp=0.0;
	        if(temp>255.0) temp=255.0;
	        convG[(i*IMG_WIDTH)+j]=(UINT8)temp;

                temp=0;
                temp += (PSF[0] * (FLOAT)B[((i-1)*IMG_WIDTH)+j-1]);
                temp += (PSF[1] * (FLOAT)B[((i-1)*IMG_WIDTH)+j]);
                temp += (PSF[2] * (FLOAT)B[((i-1)*IMG_WIDTH)+j+1]);
                temp += (PSF[3] * (FLOAT)B[((i)*IMG_WIDTH)+j-1]);
                temp += (PSF[4] * (FLOAT)B[((i)*IMG_WIDTH)+j]);
                temp += (PSF[5] * (FLOAT)B[((i)*IMG_WIDTH)+j+1]);
                temp += (PSF[6] * (FLOAT)B[((i+1)*IMG_WIDTH)+j-1]);
                temp += (PSF[7] * (FLOAT)B[((i+1)*IMG_WIDTH)+j]);
                temp += (PSF[8] * (FLOAT)B[((i+1)*IMG_WIDTH)+j+1]);
	        if(temp<0.0) temp=0.0;
	        if(temp>255.0) temp=255.0;
	        convB[(i*IMG_WIDTH)+j]=(UINT8)temp;
            }
        }

    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec  + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("stop test at %lf for %d frames, fps=%lf, pps=%lf\n\n", fnow-fstart, ITERATIONS, ITERATIONS/(fnow-fstart), (ITERATIONS*IMG_HEIGHT*IMG_WIDTH)/(fnow-fstart));

    rc=write(fdout, (void *)header, 21);

#ifdef FAST_IO

    // create in memory copy from input by channel
    for(i=0, pixel=0; i<IMG_HEIGHT*IMG_WIDTH; i++, pixel+=3)
    {
        RGB[pixel+0]=convR[i];
        RGB[pixel+1]=convG[i];
        RGB[pixel+2]=convB[i];
    }

    bytesWritten=0;
    bytesLeft=IMG_HEIGHT*IMG_WIDTH*3;
    writecnt=0;
    printf("START: write %d, bytesWritten=%d, bytesLeft=%d\n", writecnt, bytesWritten, bytesLeft);

    // Write RGB data in large chunks, requesting all at once and writing residual
    do
    {
        bytesWritten=write(fdout, (void *)&RGB[bytesWritten], bytesLeft);
        bytesLeft -= bytesWritten;
        writecnt++;

        printf("write %d, bytesWritten=%d, bytesLeft=%d\n", writecnt, bytesWritten, bytesLeft);

    } while((bytesLeft > 0) && (writecnt < 3));

    printf("END: write %d, bytesWritten=%d, bytesLeft=%d\n", writecnt, bytesWritten, bytesLeft);

#else
    // Write RGB data - very slow 1 byte at a time!
    for(i=0; i<IMG_HEIGHT*IMG_WIDTH; i++)
    {
        rc=write(fdout, (void *)&convR[i], 1);
        rc=write(fdout, (void *)&convG[i], 1);
        rc=write(fdout, (void *)&convB[i], 1);
    }
#endif


    close(fdin);
    close(fdout);
 
}
