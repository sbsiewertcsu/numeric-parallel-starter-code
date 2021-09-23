#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

#include <semaphore.h>

#define COUNT  3
#define SUB_COUNT  10

#define TOTAL_COUNT (COUNT*SUB_COUNT)

typedef struct
{
    int threadIdx;
} threadParams_t;


// POSIX thread declarations and scheduling attributes
//
pthread_t threads[2];
threadParams_t threadParams[2];

// global semaphores
sem_t incSem, decSem, doneSem;

// Unsafe global
int gsum=0;

void *incThread(void *threadp)
{
    int i, j;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    for(i=0; i<COUNT; i++)
    {
	printf("Waiting on request to increment\n");

	sem_wait(&incSem);

	printf("Proceeding to increment\n");
        for(j=0; j<SUB_COUNT; j++)
        {
            gsum++;
            printf("Increment thread idx=%d, gsum=%d\n", threadParams->threadIdx, gsum);
        }

	printf("Done with increment\n");
	sem_post(&doneSem);
    }

    sem_post(&doneSem);
}


void *decThread(void *threadp)
{
    int i, j;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    for(i=0; i<COUNT; i++)
    {
	printf("Waiting on request to decrement\n");

	sem_wait(&decSem);

	printf("Proceeding to decrement\n");
        for(j=0; j<SUB_COUNT; j++)
        {
            gsum--;
            printf("Decrement thread idx=%d, gsum=%d\n", threadParams->threadIdx, gsum);
        }

	printf("Done with decrement\n");
	sem_post(&doneSem);
    }

    sem_post(&doneSem);
}


int main (int argc, char *argv[])
{
   int rc;
   int i=0;

   if (sem_init (&incSem, 0, 0)) { printf ("Failed to initialize increment semaphore\n"); exit (-1); }
   if (sem_init (&decSem, 0, 0)) { printf ("Failed to initialize decriment semaphore\n"); exit (-1); }
   if (sem_init (&doneSem, 0, 0)) { printf ("Failed to initialize done semaphore\n"); exit (-1); }

   threadParams[i].threadIdx=i;
   pthread_create(&threads[i],   // pointer to thread descriptor
                  (void *)0,     // use default attributes
                  incThread, // thread function entry point
                  (void *)&(threadParams[i]) // parameters to pass in
                 );
   i++;

   threadParams[i].threadIdx=i;

   // attributes are default SMP, CFS scheduler
   pthread_create(&threads[i], (void *)0, decThread, (void *)&(threadParams[i]));

    for(i=0; i<=COUNT; i++)
    {
	    sem_post(&incSem);
	    sem_wait(&doneSem);

	    sem_post(&decSem);
	    sem_wait(&doneSem);
    }

   for(i=0; i<2; i++)
     pthread_join(threads[i], NULL);

   printf("TEST COMPLETE\n");
}
