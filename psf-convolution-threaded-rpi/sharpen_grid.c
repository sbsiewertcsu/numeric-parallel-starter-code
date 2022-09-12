#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>


#define IMG_HEIGHT (3000)
#define IMG_WIDTH (4000)

// Smaller, same Aspect Ratio test option
//#define IMG_HEIGHT (300)
//#define IMG_WIDTH (400)


#define NUM_ROW_THREADS (3)
#define NUM_COL_THREADS (4)

#define IMG_H_SLICE (IMG_HEIGHT/NUM_ROW_THREADS)
#define IMG_W_SLICE (IMG_WIDTH/NUM_COL_THREADS)

#define ITERATIONS (1000)

typedef double FLOAT;

pthread_t threads[NUM_ROW_THREADS*NUM_COL_THREADS];

typedef struct _threadArgs
{
    int thread_idx;
    int i;
    int j;
    int h;
    int w;
} threadArgsType;

threadArgsType threadarg[NUM_ROW_THREADS*NUM_COL_THREADS];
pthread_attr_t fifo_sched_attr;
pthread_attr_t orig_sched_attr;
struct sched_param fifo_param;

typedef unsigned int UINT32;
typedef unsigned long long int UINT64;
typedef unsigned char UINT8;

// PPM Edge Enhancement Code
UINT8 header[22];
UINT8 R[IMG_HEIGHT][IMG_WIDTH];
UINT8 G[IMG_HEIGHT][IMG_WIDTH];
UINT8 B[IMG_HEIGHT][IMG_WIDTH];
UINT8 convR[IMG_HEIGHT][IMG_WIDTH];
UINT8 convG[IMG_HEIGHT][IMG_WIDTH];
UINT8 convB[IMG_HEIGHT][IMG_WIDTH];

#define K 4.0

FLOAT PSF[9] = {-K/8.0, -K/8.0, -K/8.0, -K/8.0, K+1.0, -K/8.0, -K/8.0, -K/8.0, -K/8.0};
//FLOAT PSF[9] = {-K/80.0, -K/80.0, -K/80.0, -K/80.0, K+10.0, -K/80.0, -K/80.0, -K/80.0, -K/80.0};


void *sharpen_thread(void *threadptr)
{
    threadArgsType thargs=*((threadArgsType *)threadptr);
    int i=thargs.i;
    int j=thargs.j;
    FLOAT temp=0;

    //printf("i=%d, j=%d, h=%d, w=%d\n", thargs.i, thargs.j, thargs.h, thargs.w);

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
    int fdin, fdout, bytesRead=0, bytesLeft, i, j, idx, jdx, rc;
    unsigned int thread_idx;
    FLOAT fnow, fstart;
    int runs=0;
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

    //printf("header = %s\n", header); 

    // Read RGB data - very slow 1 byte at a time!
    for(i=0; i<IMG_HEIGHT; i++)
    {
        for(j=0; j<IMG_WIDTH; j++)
        {
            if((rc=read(fdin, (void *)&R[i][j], 1)) < 0)
		    perror("read PPM");
	    else
		    convR[i][j]=R[i][j];
            if((rc=read(fdin, (void *)&G[i][j], 1)) < 0)
		    perror("read PPM");
	    else
		    convG[i][j]=G[i][j];
            if((rc=read(fdin, (void *)&B[i][j], 1)) < 0)
		    perror("read PPM");
	    else
		    convB[i][j]=B[i][j];
        }
    }
    printf("source file %s read\n", argv[1]);
    close(fdin);


    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    fstart = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("start test at %lf\n", fnow - fstart);

    for(runs=0; runs < ITERATIONS; runs++)
    {

        for(thread_idx=0; thread_idx<(NUM_ROW_THREADS*NUM_COL_THREADS); thread_idx++)
        {

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

            //printf("idx=%d, jdx=%d\n", idx, jdx);

            threadarg[thread_idx].i=idx;      
            threadarg[thread_idx].h=IMG_H_SLICE-1;        
            threadarg[thread_idx].j=jdx;        
            threadarg[thread_idx].w=IMG_W_SLICE-1;

            //printf("create thread_idx=%d\n", thread_idx);    
            pthread_create(&threads[thread_idx], (void *)0, sharpen_thread, (void *)&threadarg[thread_idx]);

        }

        for(thread_idx=0; thread_idx<(NUM_ROW_THREADS*NUM_COL_THREADS); thread_idx++)
        {
            //printf("join thread_idx=%d\n", thread_idx);    
            if((pthread_join(threads[thread_idx], (void **)0)) < 0)
                perror("pthread_join");
        }

        //printf("frame %d completed\n", runs);

    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (FLOAT)now.tv_sec + (FLOAT)now.tv_nsec / 1000000000.0;
    printf("Completed test at %lf for %d frames and %lf FPS\n", fnow - fstart, runs, (FLOAT)runs/(fnow-fstart));

    printf("starting sink file %s write\n", argv[2]);
    if((rc=write(fdout, (void *)header, 21)) < 0) perror("read PPM");

    // Write RGB data - very slow 1 byte at a time!
    for(i=0; i<IMG_HEIGHT; i++)
    {
        for(j=0; j<IMG_WIDTH; j+=1)
        {
            if((rc=write(fdout, (void *)&convR[i][j], 1)) < 0) perror("read PPM");
            if((rc=write(fdout, (void *)&convG[i][j], 1)) < 0) perror("read PPM");
            if((rc=write(fdout, (void *)&convB[i][j], 1)) < 0) perror("read PPM");
        }
    }


    printf("sink file %s written\n", argv[2]);
    close(fdout);
 
}
