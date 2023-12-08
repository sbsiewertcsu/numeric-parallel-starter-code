// The series used for computation of the exponential exp(x) is defined well by a Taylor Series
//
// https://en.wikipedia.org/wiki/Taylor_seriesa
//
// For sin(x), see https://people.math.sc.edu/girardi/m142/handouts/10sTaylorPolySeries.pdf
//
// The series is simply x - (x*x*x)/3! + (x*x*x*x*x)/5! + ...
//
//  Note: n! is simple 1*2*3*4*5...(n-1)*n
//
//  Note - be careful with DOUBLE range, smallest decimal, and inf
//
//  Updated by Justin Daugherty, Dec 2023 to fix transcendental periodicity error
//

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <float.h>
#include <math.h>

int main(int argc, char* argv[])
{
  int thread_count;
  int iterations=0, idx=0, cnt=0;
  double sum, term;
  double x;

  if (argc < 4)
  {
    printf ( "usage: sinseries x <number threads> <iterations>\n " );
    exit(-1);
  }
  else
  {
    sscanf(argv[1], "%lf", &x);
    sscanf(argv[2], "%d", &thread_count);
    sscanf(argv[3], "%d", &iterations);

    //Added these two variables to make things a little easier to read
    unsigned int x_periods = (unsigned int)(x/(2.0*M_PI));
    double x_adjusted = x - (double)(x_periods * 2.0 * M_PI);

    printf("x=%lf, sin(x)=%lf, n periods = %d, adjusted x = %lf\n", x, sin(x), x_periods, x_adjusted);

    // Corrected issue with parens and goal to subtract off the
    // 2*Pi perodicity of the input
    x = x_adjusted;
  }

  term=x; sum=x;

  printf("DBL_EPSILON = %le\n", DBL_EPSILON);

// The goal of a series computation is to avoid use of the math library
// completely and to use only basic arithmetic to compute transcendental
// and polynomial functions.
//
// Note that you need to modify the pragma to avoid data corruption for
// variables that are read & write
//
// This can be done by careful specification of what is shared, what is
// not, and use of reduction.
//
// E.g.,
// #pragma omp parallel for num_threads(thread_count) default(none) reduction(+:sum) private(idx, term, cnt) shared(x, iterations)
//
// Alternatively, you can re-write this as a function like
// hello_openmp/piseriesompfunct.c
//
#pragma omp parallel for reduction(+:sum) num_threads(thread_count)
  for(idx=2; idx < iterations*2; idx=idx+2) 
  {
    term = -term * (x*x)/ ((double)(idx+1)*idx);

    sum += term; cnt++;
  }

  printf ( "C math library sin(x)  =%15.14lf\n", sin(x));
  printf ( "Series sin(x) =%15.14lf, iterations=%d, error=%le\n", sum, cnt, fabs(sum - sin(x)));

}

