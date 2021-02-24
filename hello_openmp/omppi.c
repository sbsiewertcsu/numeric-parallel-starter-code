#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


int main(int argc, char *argv[])
{
  int thread_count=1, k;
  double factor = 1.0;
  double sum = 0.0;
  int n=500000000;
  struct timespec start, stop;
  double fstart, fstop;


  if(argc < 2 || argc > 3)
      printf("usage: hello_omp <number threads> [iterations=500000000]\n");
  else if (argc == 2)
  {
      sscanf(argv[1], "%d", &thread_count);
  }
  else if (argc == 3)
  {
      sscanf(argv[1], "%d", &thread_count);
      sscanf(argv[2], "%d", &n);
  }

  // This is a thread parallel for loop block using OpenMP
  //
  // Here we carefully specify what variables are private (copy per thread) and which
  // are shared.  By default all would be private (copies), so we first indicate that
  // we don't want the default and then provide specific instructions.
  //

  clock_gettime(CLOCK_MONOTONIC, &start); fstart=(double)start.tv_sec + ((double)start.tv_nsec/1000000000.0);

#pragma omp parallel for num_threads(thread_count) default(none) reduction(+:sum) private(k, factor) shared(n)
  for(k=0; k < n; k++)
  {
          if(k % 2 == 0) // even k
              factor = 1.0;
          else
              factor = -1.0;

          sum += factor/(2.0*k+1.0);
  }

  clock_gettime(CLOCK_MONOTONIC, &stop); fstop=(double)stop.tv_sec + ((double)stop.tv_nsec/1000000000.0);

  printf("pi_approx=%15.14lf computer in %lf seconds\n", 4.0*sum, (fstop-fstart));

  return 0;
}
