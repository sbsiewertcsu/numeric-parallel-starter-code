// Basic OpenMP Fibonacci sequence generator that uses OpenMP for multi-thread speed-up
//
// Sam Siewert - 2/24/2021
//
#include <stdio.h>
#include <time.h>
#include <omp.h>

unsigned int fibc(unsigned int n)
{
    unsigned int f=0, f0=0, f1=1, i=1;

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

int main(int argc, char *argv[])
{
  int iterations=10000000;
  unsigned int g_fibcnt=32, g_fibnum;
  int thread_count=2;

  struct timespec start, end;
  double fstart, fend;

  double g_fibsum=0.0;

  if(argc == 1)
  {
    printf("will add up fib(32)/%lf for %d iterations\n", (double)iterations, iterations);
  }
  else if (argc == 2)
  {
      sscanf(argv[1], "%d", &iterations);
      printf("will add up fib(32)/%lf for %d iterations\n", (double)iterations, iterations);
  }
  else if (argc == 3)
  {
      sscanf(argv[1], "%d", &iterations);
      sscanf(argv[2], "%u", &g_fibcnt);
      printf("will add up fib(%u)/%lf for %d iterations\n", g_fibcnt, (double)iterations, iterations);
  }


  clock_gettime(CLOCK_MONOTONIC, &start);
  // parallel BEGIN

//# pragma omp parallel for num_threads(thread_count) schedule(static,1)
//# pragma omp parallel for num_threads(thread_count) reduction(+: g_fibsum) schedule(static,1000000)
# pragma omp parallel for num_threads(thread_count) reduction(+: g_fibsum)
  for(int idx=0; idx < iterations; idx++)
  {
      unsigned int fibcnt=g_fibcnt;
      unsigned int fibnum;

      fibnum=fibc(fibcnt);
      g_fibnum=fibnum;

//#     pragma omp critical
      g_fibsum += (double)fibnum / (double)iterations;

      //printf("fibnum = %u for thread %d\n", fibnum, omp_get_thread_num());
  }

  // parallel END
  clock_gettime(CLOCK_MONOTONIC, &end);


  fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
  fend=end.tv_sec + (end.tv_nsec / 1000000000.0);

  printf("fibsum = %lf (%u) in %lf seconds with threads=%d\n", g_fibsum, g_fibnum, fend-fstart, thread_count);
}
