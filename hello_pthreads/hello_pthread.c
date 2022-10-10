#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#define NUM_THREADS 8

typedef struct
{
    int threadIdx;
    int thread_count;
} threadParams_t;


// POSIX thread declarations and scheduling attributes
//
pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];


void *Hello_thread(void *threadp);

int main(int argc, char *argv[])
{
    int thread_count=NUM_THREADS, i;

    if(argc < 2)
        printf("usage: hello_omp <number threads>\n");
    else
    {
        sscanf(argv[1], "%d", &thread_count);
    }

   for(i=0; i < thread_count; i++)
   {
       threadParams[i].threadIdx=i;
       threadParams[i].thread_count=thread_count;

       pthread_create(&threads[i],   // pointer to thread descriptor
                      (void *)0,     // use default attributes
                      Hello_thread, // thread function entry point
                      (void *)&(threadParams[i]) // parameters to pass in
                     );

   }

   for(i=0;i<NUM_THREADS;i++)
       pthread_join(threads[i], NULL);

    return 0;
}

void *Hello_thread(void *threadp)
{
    threadParams_t *threadParams = (threadParams_t *)threadp;
    int my_rank = threadParams->threadIdx;
    int thread_count = threadParams->thread_count;


    printf("Hello from Pthread thread %d of %d\n", my_rank, thread_count);

    return((void *)0);
}
