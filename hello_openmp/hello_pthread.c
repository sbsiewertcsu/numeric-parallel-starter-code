#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>
#include <time.h>

#define MAX_THREADS (8192)
#define ERROR (-1)

typedef struct
{
    int threadIdx;
    int threadCount;
} threadParams_t;

// set up some resources for threads in this example with a simple array
pthread_t threads[MAX_THREADS];
threadParams_t threadParams[MAX_THREADS];

void *Hello_thread(void *threadp);


int main(int argc, char *argv[])
{
    int thread_count=1, idx;

    if(argc < 2)
    {
        printf("usage: hello_omp <number threads>\n");
        exit(ERROR);
    }
    else
    {
        sscanf(argv[1], "%d", &thread_count);
        if(thread_count > MAX_THREADS) 
        {
            printf("ERROR: MAX_THREADS = %d, cannot create %d threads due to resource constraints\n", MAX_THREADS, thread_count);
            exit(ERROR);
        }
    }


    printf("Main program thread will now create all threads requested...\n");

    // Create N threads that run Hello_thread function concurrently (in parallel)
    for(idx=0; idx < thread_count; idx++)
    {
        threadParams[idx].threadIdx=idx;
        threadParams[idx].threadCount=thread_count;

        pthread_create(&threads[idx],                  // pointer to POSIX thread descriptor
                       (void *)0,                      // use DEFAULT POSIX thread attributes - e.g., scheduling policy is default, etc.
                       Hello_thread,                   // function entry point for each thread to execute
                       (void *)&(threadParams[idx])    // parameters to pass to each thread
                       );
    }


    // Wait for all threads to complete before proceeding
    for(idx=0; idx < thread_count; idx++)
        pthread_join(threads[idx], NULL);

    printf("All threads are done, main program proceeding to exit\n");

    return 0;
}


void *Hello_thread(void *threadp)
{
    threadParams_t *threadParams = (threadParams_t *)threadp;
    int my_rank=ERROR, thread_count=ERROR;
    struct timespec current_time;

    // get the current monotonic system time
    //clock_gettime(CLOCK_REALTIME, &current_time);
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    my_rank = threadParams->threadIdx;
    thread_count = threadParams->threadCount;

    printf("Hello from POSIX thread %d of %d @ secs=%ld and nanseconds=%ld\n", my_rank, thread_count, current_time.tv_sec, current_time.tv_nsec);

    pthread_exit(NULL);
}
