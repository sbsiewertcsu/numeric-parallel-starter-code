#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

#define NUM_THREADS 10

typedef struct
{
int threadIdx;
int start;
int end;
int sum;
} threadParams_t;

// POSIX thread declarations and scheduling attributes
pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];

int globalSum = 0;

void *helloThread(void *threadp)
{
threadParams_t *threadParams = (threadParams_t *)threadp;

for (int i = threadParams->start; i <= threadParams->end; i++) {
threadParams->sum += i;
}

printf("Thread %d: Counter sum[%d...%d]=%d\n", threadParams->threadIdx, threadParams->start, threadParams->end, threadParams->sum);

return ((void *)0);
}

int main (int argc, char *argv[])
{
for (int i = 0; i < NUM_THREADS; i++) {
threadParams[i].threadIdx = i;
threadParams[i].start = i * 10;
threadParams[i].end = threadParams[i].start + 9;
threadParams[i].sum = 0;

//create the pthreads
pthread_create(&threads[i], NULL, helloThread, &threadParams[i]);
}

for (int idx = 0; idx < NUM_THREADS; idx++)
pthread_join(threads[idx], NULL);

for (int j = 0; j < NUM_THREADS; j++)
globalSum += threadParams[j].sum;

printf("gsum[0...99]=%d, check=%d\n", globalSum, 4950);

printf("TEST COMPLETE\n");

return 0;
}
