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


void *helloThread(void *threadp)
{
    threadParams_t *threadParams = (threadParams_t *)threadp;

    printf("Thread %d: hello\n", threadParams->threadIdx);

    return((void *)0);
}


int main (int argc, char *argv[])
{
   int idx;

   for(idx=0; idx < NUM_THREADS; idx++)
   {
       threadParams[idx].threadIdx=idx;

       pthread_create(&threads[idx],   // pointer to thread descriptor
                      (void *)0,     // use default attributes
                      helloThread, // thread function entry point
                      (void *)&(threadParams[idx]) // parameters to pass in
                     );

   }

   for(idx=0;idx<NUM_THREADS;idx++)
       pthread_join(threads[idx], NULL);

   printf("TEST COMPLETE\n");
}
