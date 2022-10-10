#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sched.h>

#define MAX_NUM_THREADS 128

typedef struct
{
    int threadIdx;
    int n;
    double local_sum;
    int thread_count;
} threadParams_t;


// POSIX thread declarations and scheduling attributes
//
pthread_t threads[MAX_NUM_THREADS];
threadParams_t threadParams[MAX_NUM_THREADS];


void *seriesThread(void *threadp)
{
    int k;
    double sum=0.0, num=1.0;
    threadParams_t *threadParams = (threadParams_t *)threadp;
    int n = threadParams->n;
    int length = n / threadParams->thread_count;
    int residual = n % threadParams->thread_count;
    int iterations;

    if(threadParams->threadIdx == (threadParams->thread_count -1))
        iterations = length + residual;
    else
        iterations = length;

    //printf("Thread %d of %d, length=%d, residual=%d, iterations=%d\n",
    //       threadParams->threadIdx, threadParams->thread_count, length, residual, iterations);

    for(k=((threadParams->threadIdx)*length); k < ((threadParams->threadIdx)*length)+iterations; k++)
    {
        num=1.0/(1.0+2.0*k);

        if(k % 2 == 0) // even k
            sum += num;
        else
            sum -= num;
    }

    threadParams->local_sum=sum;

    printf("thread %d: thread sum = %016.15lf\n", threadParams->threadIdx, threadParams->local_sum);

    return((void *)0);
}


int main(int argc, char *argv[])
{
  int thread_count=1, i=0;
  double sum = 0.0;
  int n=500000000;
  struct timespec start, stop;
  double fstart, fstop;


  if (argc == 3)
  {
      sscanf(argv[1], "%d", &thread_count);
      sscanf(argv[2], "%d", &n);
  }
  else if (argc == 2)
  {
      sscanf(argv[1], "%d", &thread_count);
  }
  else
  {
      printf("usage: pthreadpi <number threads> [iterations=500000000]\n");
      thread_count=4;
      printf("Defaulting thread_count = %d, n = %d\n", thread_count, n);
  }

  if(thread_count > MAX_NUM_THREADS) { printf("Too many threads ...\n"); exit(-1); }

  // This is a thread parallel for loop block using Pthreads
  //
  // Here we carefully specify what variables are private (copy per thread) and which
  // are shared.  By default all would be private (copies), so we first indicate that
  // we don't want the default and then provide specific instructions.
  //

  clock_gettime(CLOCK_MONOTONIC, &start); fstart=(double)start.tv_sec + ((double)start.tv_nsec/1000000000.0);

  for(i=0; i < thread_count; i++)
  {
       threadParams[i].threadIdx=i;
       threadParams[i].n=n;
       threadParams[i].thread_count=thread_count;

       pthread_create(&threads[i],   // pointer to thread descriptor
                      (void *)0,     // use default attributes
                      seriesThread, // thread function entry point
                      (void *)&(threadParams[i]) // parameters to pass in
                     );

   }

   for(i=0;i<thread_count;i++)
   {
       pthread_join(threads[i], NULL);
       sum+=threadParams[i].local_sum;
   }

   clock_gettime(CLOCK_MONOTONIC, &stop); fstop=(double)stop.tv_sec + ((double)stop.tv_nsec/1000000000.0);

   printf("\nRESULTS:\n");
   printf("20 decimals of pi  =3.14159265358979323846\n");
   printf("C math library pi  =%15.14lf\n" , M_PI);
   printf("Madhava-Leibniz pi =%15.14lf, ppb error=%15.14lf, computed in %lf seconds\n", 4.0*sum, 1000000000.0*(M_PI-(4.0*sum )), (fstop-fstart));

   return 0;
}
