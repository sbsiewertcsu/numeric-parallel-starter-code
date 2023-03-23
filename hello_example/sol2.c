// Solution for problem 2
//
// Divide up work to sum digits with 10 threads and then sum the sums
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

#define NUM_THREADS 10
#define SUM_DIGITS_RANGE (100)  // could become argument for problem scaling, but keep it simple here

typedef struct
{
    int threadIdx;
    int start;
    int end;
    int local_sum;
} threadParams_t;

// POSIX thread declarations and scheduling attributes
pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];

int gsum = 0;
void *workerThread(void *threadp);


int main (int argc, char *argv[])
{
    for (int idx=0; idx<NUM_THREADS; idx++) 
    {
        threadParams[idx].threadIdx = idx;
        threadParams[idx].start = idx*NUM_THREADS;
        threadParams[idx].end = threadParams[idx].start+(NUM_THREADS-1);
        threadParams[idx].local_sum = 0;

        pthread_create(&threads[idx], NULL, workerThread, &threadParams[idx]);
    }

    for (int idx = 0; idx < NUM_THREADS; idx++)
        pthread_join(threads[idx], NULL);

    for (int idx=0; idx<NUM_THREADS; idx++)
        gsum += threadParams[idx].local_sum;

    printf("gsum[0...99]=%d, check=%d\n", gsum, 4950);

    return 0;
}


void *workerThread(void *threadp)
{
    threadParams_t *threadParams = (threadParams_t *)threadp;

    for (int idx=threadParams->start; idx<=threadParams->end; idx++) 
        threadParams->local_sum += idx;

    printf("Thread %d: Counter sum[%d...%d]=%d\n",
           threadParams->threadIdx, threadParams->start, threadParams->end, threadParams->local_sum);

    return ((void *)0);
}
