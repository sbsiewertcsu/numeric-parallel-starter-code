#include <stdio.h>
#include <time.h>
#include <omp.h>

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

int main(int argc, char *argv[])
{
  unsigned int g_fibcnt=64;
  unsigned long long int g_fibnum;

  struct timespec start, end;
  double fstart, fend;

  if (argc == 2)
  {
      sscanf(argv[1], "%u", &g_fibcnt);
      printf("will compute %uth Fibonacci number\n", g_fibcnt);
  }
  else
      printf("will compute %uth Fibonacci number\n", g_fibcnt);



  clock_gettime(CLOCK_MONOTONIC, &start);
  // parallel BEGIN

# pragma omp parallel num_threads(4)
  {
      unsigned int fibcnt=g_fibcnt;
      unsigned long long int fibnum;

      fibnum=fibc(fibcnt);
      printf("fibnum=%llu for thread %d\n", fibnum, omp_get_thread_num());
      g_fibnum=fibnum;
  }

  // parallel END
  clock_gettime(CLOCK_MONOTONIC, &end);


  fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
  fend=end.tv_sec + (end.tv_nsec / 1000000000.0);

  printf("g_fibnum=%llu in %lf seconds\n", g_fibnum, fend-fstart);
}
