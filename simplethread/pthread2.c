#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

#define NUM_THREADS 10000

typedef struct
{
    int threadIdx;
    int start;
    int end;
    unsigned long int mysum;
} threadParams_t;


// POSIX thread declarations and scheduling attributes
//
pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];


void *counterThread(void *threadp)
{
    int i;
    unsigned long int sum=0;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    for(i=threadParams->start; i < (threadParams->end); i++)
    {
        sum=sum+i;
    }

    threadParams->mysum = sum;

    printf("COMPLETE Thread idx=%d, sum[%08d...%08d]=%lu\n", 
           threadParams->threadIdx,
           threadParams->start, threadParams->end-1, threadParams->mysum);

    return((void *)0);
}

#define MAX_RANGE (1000000000)

int main (int argc, char *argv[])
{
   int i, numthreads=10, start, end;
   unsigned long int sum=0, check;

   if(argc == 2)
   {
       sscanf(argv[1], "%d", &numthreads);
       printf("will run with %d threads\n", numthreads);
   }
   else
   {
       printf("will run with default %d threads\n", numthreads);
   }

   // Limit threads so we divide work evenly for powers of 10 since we want to scale
   // the summing range by powers of 10 for simplicity
   if(((numthreads % 10) != 0) && (numthreads != 1))
   {
       printf("uneven division of summing range\n");
       exit(-1);
   }

   // 0 to (MAX/nthreads) - 1
   start=0;
   end=(MAX_RANGE / numthreads);

   for(i=0; i < numthreads; i++)
   {
       threadParams[i].threadIdx=i;
       threadParams[i].start=start;
       threadParams[i].end=end;

       pthread_create(&threads[i],   // pointer to thread descriptor
                      (void *)0,     // use default attributes
                      counterThread, // thread function entry point
                      (void *)&(threadParams[i]) // parameters to pass in
                     );

        start = end;
        end = end + (MAX_RANGE / numthreads);

   }

   // main thread adds the last number in 0...MAX_RANGE
   //
   // all other threads sum ranges in 0...MAX_RANGE-1
   sum = MAX_RANGE;
   check = (sum * (sum+1))/2;

   for(i=0;i<numthreads;i++)
   {
      pthread_join(threads[i], NULL);

      printf("JOIN Thread idx=%d, sum[%08d...%08d]=%lu\n", i,
             threadParams[i].start, threadParams[i].end-1, threadParams[i].mysum);

   }


   for(i=0;i<numthreads;i++)
      sum += threadParams[i].mysum;


   printf("sum of sums = %lu, check = %lu\n", sum, check);

   printf("TEST COMPLETE\n");
}
