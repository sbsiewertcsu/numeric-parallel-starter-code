
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

#define NUM_THREADS 10

typedef struct
{
int threadIdx;
} threadParams_t;


// POSIX thread declarations and scheduling attributes
pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];
int global_sum[NUM_THREADS];


void *helloThread(void *threadp)
{
threadParams_t *threadParams = (threadParams_t *)threadp;

int start = threadParams->threadIdx * 10;
int end = start + 9;
int sum = 0;
for (int i = start; i <= end; i++)
{
sum += i;
}
global_sum[threadParams->threadIdx] = sum;

printf("Thread %d: Counter sum [%d..%d] = %d]\n", threadParams->threadIdx, start, end, sum);

return((void *)0);
}



int main (int argc, char *argv[])
{
int idx;
int result_sum = 0;
// Write a Pthread program that uses 10 threads to divide work evenly to produce the sum of 0 to 99
for(idx=0; idx < NUM_THREADS; idx++)
{
threadParams[idx].threadIdx=idx;

pthread_create(&threads[idx], // pointer to thread descriptor
(void *)0, // use default attributes
helloThread, // thread function entry point
(void *)&(threadParams[idx]) // parameters to pass in
);

}

for(idx=0;idx<NUM_THREADS;idx++) {
pthread_join(threads[idx], NULL);
result_sum += global_sum[idx];
}

printf("gsum[0..99]=%d, check=%d\n", result_sum, (0 + 99) * 100 / 2);
}

