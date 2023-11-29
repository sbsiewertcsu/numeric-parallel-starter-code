// Notes:
//
// https://www.mathsisfun.com/numbers/fibonacci-sequence.html
//
// The sequence is simple but has an inherent loop-carried dependency.  This makes it non-trivial to
// speed-up with parallelism.
//
// Approaches to solve this include OpenMP tasking and/or dynamic programming strategies.  Geeks for geeks has documented
// multiple coding methods including dynamic programming here:
// * https://www.geeksforgeeks.org/cpp-program-for-fibonacci-numbers/
//
// Also, the n'th fibonacci can be a very large number that overflows a 64-bit unsigned integer, another interesting
// complication of an otherwise simple series.
//
// For C++, one approach to dealing with large numbers is GMP - https://gmplib.org/ or using Boost:
// * https://www.geeksforgeeks.org/generating-large-fibonacci-numbers-using-boost-library/
// * https://codereview.stackexchange.com/questions/213101/calculate-huge-fibonacci-numbers-in-c-using-gmp-and-boostmultiprecision
//
// Large fibonacci numbers are interesting from a math theory viewpoint.  For example, "As of September 2023,
// the largest known certain Fibonacci prime is F201107, with 42029 digits. It was proved prime by Maia Karpovich
// in September 2023.[4] The largest known probable Fibonacci prime is F6530879. It was found by Ryan Propper in
// August 2022.[2] - https://en.wikipedia.org/wiki/Fibonacci_prime
//
// Here's some lists of the first 10, 100, 300, and 500 fibonacci numbers:
// * https://www.math.net/list-of-fibonacci-numbers
// * https://planetmath.org/listoffibonaccinumbers
//
// Pacheco Textbook code for fibonacci examples:
// * https://www.ecst.csuchico.edu/~sbsiewert/csci551/code/ipp2-source/ch5/
// * https://www.ecst.csuchico.edu/~sbsiewert/csci551/code/ipp2-source/ch5/omp_fib.c
// * https://www.ecst.csuchico.edu/~sbsiewert/csci551/code/ipp2-source/ch5/omp_fibo.c
// * https://www.ecst.csuchico.edu/~sbsiewert/csci551/code/ipp2-source/ch5/omp_fib_time.c
// * https://www.ecst.csuchico.edu/~sbsiewert/csci551/code/ipp2-source/ch5/omp_fib_broken1.c
// * https://www.ecst.csuchico.edu/~sbsiewert/csci551/code/ipp2-source/ch5/omp_fib_broken2.c
//
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

#define WORKERS 1

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
  unsigned int g_fibcnt=32;
  unsigned long long int g_fibnum;

  struct timespec start, end;
  double fstart, fend;

  if (argc == 2)
  {
      sscanf(argv[1], "%u", &g_fibcnt);
      printf("will compute Fibonacci of %u\n", g_fibcnt);
  }
  else
      printf("will compute Fibonacci of %u\n", g_fibcnt);

  // Program range limitation based on unsigned 64-bit results
  if(g_fibcnt > 93)
  {
      printf("Program cannot compute a fiboannci number greater than 32 - will overflow\n");
      exit(-1);
  }


  // parallel BEGIN
  clock_gettime(CLOCK_MONOTONIC, &start);

// simple pragma may appear to work, but it does not divide work, it simply replicates it
# pragma omp parallel num_threads(WORKERS)
  {
      unsigned int fibcnt=g_fibcnt;
      unsigned long long int fibnum;

      fibnum=fibc(fibcnt);
      printf("fibnum=%llu for thread %d\n", fibnum, omp_get_thread_num());
      g_fibnum=fibnum;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  // parallel END


  fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
  fend=end.tv_sec + (end.tv_nsec / 1000000000.0);

  printf("g_fibnum=%llu in %lf seconds\n", g_fibnum, fend-fstart);
}
