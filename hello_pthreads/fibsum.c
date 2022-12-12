#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define NUM_THREADS 4

typedef struct
{
    int threadIdx;
    int thread_count;
    unsigned long long local_fibnum;
} threadParams_t;


// POSIX thread declarations and scheduling attributes
//
pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];

int global_iterations=10000000;
unsigned int global_fibcnt=32;


unsigned long long int fibc(unsigned int n)
{
    unsigned long long int f=0, f0=0, f1=1, i=1;

    // the fibonacci sequence has inherent data dependencies
    do
    {
        i++;
        f = f0 + f1; /* This is f(2), f(3), f(n-1) */
        f0 = f1;
        f1 = f;
    } while(i < n);

    return f;
}


void *fib_wrapper_thread(void *threadp)
{
    threadParams_t *threadParams = (threadParams_t *)threadp;
    int my_rank = threadParams->threadIdx;
    int thread_count = threadParams->thread_count;

    threadParams->local_fibnum=fibc(global_fibcnt);
    printf("Pthread %d of %d with fib(%u)=%llu\n", my_rank, thread_count, global_fibcnt, threadParams->local_fibnum);

    return((void *)0);
}


int main(int argc, char *argv[])
{
  unsigned int g_fibcnt=32;
  unsigned long long int g_fibnum=0;
  int thread_count = NUM_THREADS, idx=0;
  double g_fibsum=0.0;

  struct timespec start, end;
  double fstart, fend;

  if (argc == 2)
  {
      sscanf(argv[1], "%u", &g_fibcnt);
      printf("will compute Fibonacci of %u\n", g_fibcnt);
  }
  else
      printf("will compute Fibonacci of %u\n", g_fibcnt);



  clock_gettime(CLOCK_MONOTONIC, &start);

  // parallel BEGIN

   for(idx=0; idx < thread_count; idx++)
   {
       threadParams[idx].threadIdx=idx;
       threadParams[idx].thread_count=thread_count;

       pthread_create(&threads[idx],   // pointer to thread descriptor
                      (void *)0,     // use default attributes
                      fib_wrapper_thread, // thread function entry point
                      (void *)&(threadParams[idx]) // parameters to pass in
                     );

   }

   for(idx=0;idx<NUM_THREADS;idx++)
   {
       pthread_join(threads[idx], NULL);

       // Take the fibonacci from any of thread results since they all should be the same
       g_fibnum = threadParams[idx].local_fibnum;

       g_fibsum += g_fibnum;
   }

  // parallel END

  clock_gettime(CLOCK_MONOTONIC, &end);

  fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
  fend=end.tv_sec + (end.tv_nsec / 1000000000.0);

  printf("g_fibnum=%llu with fibsum of %lf (%lf) in %lf seconds\n", g_fibnum, g_fibsum, (g_fibsum/NUM_THREADS), fend-fstart);
}
