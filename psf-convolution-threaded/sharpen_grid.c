#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>


// See notes on images in Week-3-1 for tips on thread gridding and note that in this example, tiles
// are used for a 4:3 aspect ratio image resolution.
//
// This could be simplified into horizontal slices (rows) of the 1D (or 2D version) of the image array.
//

#define IMG_HEIGHT (3000)
#define IMG_WIDTH (4000)

//#define IMG_HEIGHT (300)
//#define IMG_WIDTH (400)


// Scheme to index by simple row threads, ignoring number of columns
//#define NUM_ROW_THREADS (1)
//#define NUM_COL_THREADS (1)

//#define NUM_ROW_THREADS (2)
//#define NUM_COL_THREADS (1)

//#define NUM_ROW_THREADS (3)
//#define NUM_COL_THREADS (1)

//#define NUM_ROW_THREADS (4)
//#define NUM_COL_THREADS (1)


// Row and column threads to process in tiles for 4:3 aspect ratio
#define NUM_ROW_THREADS (3)
#define NUM_COL_THREADS (4)

//#define NUM_ROW_THREADS (6)
//#define NUM_COL_THREADS (8)

//#define NUM_ROW_THREADS (9)
//#define NUM_COL_THREADS (12)

//#define NUM_ROW_THREADS (12)
//#define NUM_COL_THREADS (16)

//#define NUM_ROW_THREADS (15)
//#define NUM_COL_THREADS (20)

//#define NUM_ROW_THREADS (18)
//#define NUM_COL_THREADS (24)

//#define NUM_ROW_THREADS (21)
//#define NUM_COL_THREADS (28)

//#define NUM_ROW_THREADS (24)
//#define NUM_COL_THREADS (32)


#define IMG_H_SLICE (IMG_HEIGHT/NUM_ROW_THREADS)
#define IMG_W_SLICE (IMG_WIDTH/NUM_COL_THREADS)

#define SHARPEN_GRID_ITERATIONS (90)  // Number of times threads are created to process one image

#define FAST_IO

typedef double FLOAT;

pthread_t threads[NUM_ROW_THREADS*NUM_COL_THREADS];

typedef struct _threadArgs
{
    int thread_idx;
    int i;
    int j;
    int h;
    int w;
    int iterations;
} threadArgsType;

threadArgsType threadarg[NUM_ROW_THREADS*NUM_COL_THREADS];
pthread_attr_t fifo_sched_attr;
pthread_attr_t orig_sched_attr;
struct sched_param fifo_param;

typedef unsigned int UINT32;
typedef unsigned long long int UINT64;
typedef unsigned char UINT8;

// PPM Edge Enhancement Code in row x column format
UINT8 header[22];
UINT8 R[IMG_HEIGHT][IMG_WIDTH];
UINT8 G[IMG_HEIGHT][IMG_WIDTH];
UINT8 B[IMG_HEIGHT][IMG_WIDTH];
UINT8 convR[IMG_HEIGHT][IMG_WIDTH];
UINT8 convG[IMG_HEIGHT][IMG_WIDTH];
UINT8 convB[IMG_HEIGHT][IMG_WIDTH];

// PPM format array with RGB channels all packed together
UINT8 RGB[IMG_HEIGHT*IMG_WIDTH*3];


#define K 4.0
#define F 8.0
//#define F 80.0

FLOAT PSF[9] = {-K/F, -K/F, -K/F, -K/F, K+1.0, -K/F, -K/F, -K/F, -K/F};


void *sharpen_thread(void *threadptr)
{
    threadArgsType thargs=*((threadArgsType *)threadptr);
    int i=thargs.i;
    int j=thargs.j;
    int repeat=0;
    FLOAT temp=0;

    //printf("i=%d, j=%d, h=%d, w=%d, iter=%d\n", thargs.i, thargs.j, thargs.h, thargs.w, thargs.iterations);

    for(i=thargs.i; i<(thargs.i+thargs.h); i++)
    {
        for(j=thargs.j; j<(thargs.j+thargs.w); j++)
        {
            temp=0;
            temp += (PSF[0] * (FLOAT)R[(i-1)][j-1]);
            temp += (PSF[1] * (FLOAT)R[(i-1)][j]);
            temp += (PSF[2] * (FLOAT)R[(i-1)][j+1]);
            temp += (PSF[3] * (FLOAT)R[(i)][j-1]);
            temp += (PSF[4] * (FLOAT)R[(i)][j]);
            temp += (PSF[5] * (FLOAT)R[(i)][j+1]);
            temp += (PSF[6] * (FLOAT)R[(i+1)][j-1]);
            temp += (PSF[7] * (FLOAT)R[(i+1)][j]);
            temp += (PSF[8] * (FLOAT)R[(i+1)][j+1]);
	        if(temp<0.0) temp=0.0;
            if(temp>255.0) temp=255.0;
            convR[i][j]=(UINT8)temp;

            temp=0;
            temp += (PSF[0] * (FLOAT)G[(i-1)][j-1]);
            temp += (PSF[1] * (FLOAT)G[(i-1)][j]);
            temp += (PSF[2] * (FLOAT)G[(i-1)][j+1]);
            temp += (PSF[3] * (FLOAT)G[(i)][j-1]);
            temp += (PSF[4] * (FLOAT)G[(i)][j]);
            temp += (PSF[5] * (FLOAT)G[(i)][j+1]);
            temp += (PSF[6] * (FLOAT)G[(i+1)][j-1]);
            temp += (PSF[7] * (FLOAT)G[(i+1)][j]);
            temp += (PSF[8] * (FLOAT)G[(i+1)][j+1]);
    	    if(temp<0.0) temp=0.0;
    	    if(temp>255.0) temp=255.0;
    	    convG[i][j]=(UINT8)temp;

            temp=0;
            temp += (PSF[0] * (FLOAT)B[(i-1)][j-1]);
            temp += (PSF[1] * (FLOAT)B[(i-1)][j]);
            temp += (PSF[2] * (FLOAT)B[(i-1)][j+1]);
            temp += (PSF[3] * (FLOAT)B[(i)][j-1]);
            temp += (PSF[4] * (FLOAT)B[(i)][j]);
            temp += (PSF[5] * (FLOAT)B[(i)][j+1]);
            temp += (PSF[6] * (FLOAT)B[(i+1)][j-1]);
            temp += (PSF[7] * (FLOAT)B[(i+1)][j]);
            temp += (PSF[8] * (FLOAT)B[(i+1)][j+1]);
    	    if(temp<0.0) temp=0.0;
    	    if(temp>255.0) temp=255.0;
    	    convB[i][j]=(UINT8)temp;
        }
    }

    pthread_exit((void **)0);
}


int main(int argc, char *argv[])
{
    int fdin, fdout, bytesRead=0, bytesWritten=0, bytesLeft, i, j, idx, jdx, pixel, readcnt, writecnt;
    UINT64 microsecs=0, millisecs=0;
    unsigned int thread_idx;
    FLOAT temp, fnow, fstart;
    int runs=0, rc;
    struct timespec now, start;

    clock_gettime(CLOCK_MONOTONIC, &start);
    fstart = (FLOAT)start.tv_sec + (FLOAT)start.tv_nsec / 1000000000.0;
    
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

    do
    {
        //printf("bytesRead=%d, bytesLeft=%d\n", bytesRead, bytesLeft);
        bytesRead=read(fdin, (void *)header, bytesLeft);
        bytesLeft -= bytesRead;

    } while(bytesLeft > 0);

    header[21]='\0';

    printf("header = %s\n", header); 


#ifdef FAST_IO
    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    fstart = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("\nstart test input at %lf\n", fnow - fstart);

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
    for(i=0, pixel=0; i<IMG_HEIGHT; i++)
    {
        for(j=0; j<IMG_WIDTH; j++)
        {
            R[i][j]=RGB[pixel+0]; convR[i][j]=R[i][j];
            G[i][j]=RGB[pixel+1]; convG[i][j]=G[i][j];
            B[i][j]=RGB[pixel+2]; convB[i][j]=B[i][j];
            pixel+=3;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("\ncompleted test input at %lf\n", fnow - fstart);
#else

    // Read RGB data - very slow 1 byte at a time!
    for(i=0; i<IMG_HEIGHT; i++)
    {
        for(j=0; j<IMG_WIDTH; j++)
        {
            rc=read(fdin, (void *)&R[i][j], 1); convR[i][j]=R[i][j];
            rc=read(fdin, (void *)&G[i][j], 1); convG[i][j]=G[i][j];
            rc=read(fdin, (void *)&B[i][j], 1); convB[i][j]=B[i][j];
        }
    }
#endif


    printf("source file %s read\n", argv[1]);
    close(fdin);


    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    fstart = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("\nstart test at %lf\n", fnow - fstart);

    for(runs=0; runs < SHARPEN_GRID_ITERATIONS; runs++)
    {

        for(thread_idx=0; thread_idx<(NUM_ROW_THREADS*NUM_COL_THREADS); thread_idx++)
        {

            // Simplified threading can just break the image into full rows for each thread
            //
            // E.g. Assuming an even number of rows 2n (most common), then I can divide into 
            //      1, 2, 4, 8, ..., 2n threads and rows up to one row per thread without needing
            //      any division into columns.


            // True tiled image processing per thread is difficult and relys upon an aspect ratio where
            // tiles fit evenly into that aspect ratio.
            
            // hard coded for 4 x 3 threads, could generalize to any 4:3 aspect ratio
            //
            // Adapting this code for array indexing for any number of threads is left as an exercise for
            // students.
            
#if (NUM_ROW_THREADS == 3) && (NUM_COL_THREADS == 4)
            if(thread_idx == 0) {idx=1; jdx=1;}
            if(thread_idx == 1) {idx=1; jdx=(thread_idx*(IMG_W_SLICE-1));}
            if(thread_idx == 2) {idx=1; jdx=(thread_idx*(IMG_W_SLICE-1));}
            if(thread_idx == 3) {idx=1; jdx=(thread_idx*(IMG_W_SLICE-1));}

            if(thread_idx == 4) {idx=IMG_H_SLICE; jdx=(thread_idx*(IMG_W_SLICE-1));}
            if(thread_idx == 5) {idx=IMG_H_SLICE; jdx=(thread_idx*(IMG_W_SLICE-1));}
            if(thread_idx == 6) {idx=IMG_H_SLICE; jdx=(thread_idx*(IMG_W_SLICE-1));}
            if(thread_idx == 7) {idx=IMG_H_SLICE; jdx=(thread_idx*(IMG_W_SLICE-1));}

            if(thread_idx == 8) {idx=(2*(IMG_H_SLICE-1)); jdx=(thread_idx*(IMG_W_SLICE-1));}
            if(thread_idx == 9) {idx=(2*(IMG_H_SLICE-1)); jdx=(thread_idx*(IMG_W_SLICE-1));}
            if(thread_idx == 10) {idx=(2*(IMG_H_SLICE-1)); jdx=(thread_idx*(IMG_W_SLICE-1));}
            if(thread_idx == 11) {idx=(2*(IMG_H_SLICE-1)); jdx=(thread_idx*(IMG_W_SLICE-1));}
#else
#error "Code must be re-written for thread indexing into array for thread 4:3 thread gridding and 12 threads"
#endif

            //printf("idx=%d, jdx=%d\n", idx, jdx);

            threadarg[thread_idx].i=idx;      
            threadarg[thread_idx].h=IMG_H_SLICE-1;        
            threadarg[thread_idx].j=jdx;        
            threadarg[thread_idx].w=IMG_W_SLICE-1;

            //threadarg[thread_idx].iterations=THREAD_ITERATIONS;

            //printf("create thread_idx=%d\n", thread_idx);    
            rc=pthread_create(&threads[thread_idx], (void *)0, sharpen_thread, (void *)&threadarg[thread_idx]);
            if(rc < 0)
            {
                    perror("pthread_create");
                    exit(-1);
            }
            //else
            //{
            //    printf("create thread_idx=%d\n", thread_idx);    
            //}
        }

        // Join in same order created, but opposite order is a bit safer since thread created last will
        // likely be slowest to join.
        //
        for(thread_idx=(NUM_ROW_THREADS*NUM_COL_THREADS); thread_idx > 0; thread_idx--)
        //for(thread_idx=0; thread_idx<(NUM_ROW_THREADS*NUM_COL_THREADS); thread_idx++)
        {
            //printf("join thread_idx=%d\n", thread_idx-1);    

            if((pthread_join(threads[thread_idx-1], (void **)0)) < 0)
            {
                perror("pthread_join");
                exit(-1);
            }
        }

        //printf("create run=%d ", runs);

    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("\nCompleted test at %lf for %d create-to-join and %lf FPS\n\n", fnow - fstart, runs, (FLOAT)SHARPEN_GRID_ITERATIONS/(fnow-fstart));

    printf("Starting output file %s write\n", argv[2]);
    rc=write(fdout, (void *)header, 21);


#ifdef FAST_IO
    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    fstart = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("\nstart test input at %lf\n", fnow - fstart);


    // create in memory copy from input by channel
    for(i=0, pixel=0; i<IMG_HEIGHT; i++)
    {
        for(j=0; j<IMG_WIDTH; j++)
        {
            RGB[pixel+0]=convR[i][j];
            RGB[pixel+1]=convG[i][j];
            RGB[pixel+2]=convB[i][j];
            pixel+=3;
        }
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

    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("\ncompleted test input at %lf\n", fnow - fstart);

#else

    // Write RGB data - very slow 1 byte at a time!
    for(i=0; i<IMG_HEIGHT; i++)
    {
        for(j=0; j<IMG_WIDTH; j+=1)
        {
            rc=write(fdout, (void *)&convR[i][j], 1);
            rc=write(fdout, (void *)&convG[i][j], 1);
            rc=write(fdout, (void *)&convB[i][j], 1);
        }
    }

#endif

    printf("Output file %s written\n", argv[2]);
    close(fdout);
 
}
