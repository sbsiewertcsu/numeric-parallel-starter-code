#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>


// Note that this starter code example (and all starter code in CSCI 551) should be modified for problem to be solved,
// well commented and explained, and you should "make it your own".
//
// You may be asked to walk-through and explain your code solution, so if it is adapted from starter code,
// make sure you don't cut-and-paste or re-use whole-sale without understanding.
//
// Generally starter code has been tested and does not intentionally include bugs, but it is also not "solution"
// code, so assume that you must adapt, improve, and even totally re-code (re-factor) for your solution.
//
// If you have questions about starter code, come to office hours, bring quesitons to class, and the instructor/author of this code
// will gladly walk-through and explain what is here.
//
// C or C++ is equally acceptable for CSCI 551, despite the fact that most starter code is simple C
//

#define COUNT  1000000ULL
//#define COUNT  1000ULL

#define NUM_THREADS (10)

typedef struct
{
    int threadIdx;
} threadParams_t;


// POSIX thread declarations and scheduling attributes
pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];

// Thread specific globals - static initializer must be updated for sub-ranges
long int gsum[NUM_THREADS];

void *sumThread(void *threadp)
{
    int idx;
    long int startval, endval, i;
    threadParams_t *threadParams = (threadParams_t *)threadp;
    idx = threadParams->threadIdx;

    startval=(idx*(COUNT/NUM_THREADS))+1;
    endval=((idx*(COUNT/NUM_THREADS))+(COUNT/NUM_THREADS));

    for(i=startval; i<=endval; i++)
    {
        // Simply sum the digits for this particular thread's assigned range
        gsum[idx] = gsum[idx] + i;

        // UNCOMMENTING THIS LINE - will give you tons of output and a full trace of each thread's progress
        // Please do turn it on to see behavior, but for production code, we normally would not leave this
        // type of tracing in the code since it will slow it down.
        //printf("Increment %ld for thread idx=%d, gsum=%ld\n", i, idx, gsum[idx]);
    }
    printf("thread idx=%d for range %ld to %ld, gsum=%ld, should be %ld\n", idx, startval, endval, gsum[idx], 
           ((endval*(endval+1))/2) - (((startval-1)*startval)/2) );
}

int main (int argc, char *argv[])
{
   int rc;
   long int i=0;
   long int testsum=0, gsumall=0;
   long int startval, endval;


   for(i=0; i<NUM_THREADS; i++)
   {
      threadParams[i].threadIdx=i;
      pthread_create(&threads[i], (void *)0, sumThread, (void *)&(threadParams[i]));
   }

   for(i=0; i<NUM_THREADS; i++)
     pthread_join(threads[i], NULL);

   for(i=0; i<NUM_THREADS; i++)
   {
       startval=(i*(COUNT/NUM_THREADS))+1;
       endval=((i*(COUNT/NUM_THREADS))+(COUNT/NUM_THREADS));
       printf("Thread %ld COMPLETE: gsum[%ld]=%ld for range %ld to %ld\n", i, i, gsum[i], startval, endval);
       gsumall += gsum[i];
   }
   printf("TEST COMPLETE: gsumall=%ld\n", gsumall);

   // Verfiy with single thread version and (n*(n+1))/2
   for(i=0; i<=COUNT; i++)
       testsum = testsum + i;

   printf("TEST COMPLETE: COUNT=%llu, single thread testsum=%ld, [n[n+1]]/2=%llu\n", COUNT, testsum, (unsigned long long)((COUNT)*(COUNT+1))/2);
}
