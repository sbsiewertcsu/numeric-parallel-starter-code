// The series used for computation of the exponential exp(x) is defined well by a Taylor Series
//
// https://en.wikipedia.org/wiki/Taylor_seriesa
//
// For exp(x), see https://people.math.sc.edu/girardi/m142/handouts/10sTaylorPolySeries.pdf
//
// The series is simply 1 + x + (x*x)/2! + (x*x*x)/3! + ...
//
//  Note: n! is simple 1*2*3*4*5...(n-1)*n
//
//  Note - be careful with DOUBLE range, smallest decimal, and inf
//

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <float.h>
#include <math.h>

int main(int argc, char* argv[])
{
  int thread_count;
  int iterations=0, idx=0, jdx=0, cnt=0;
  double sum=0.0;
  double term;
  double num, fact, err, preverr=DBL_MAX;
  double x;

  if (argc < 4)
  {
    printf ( "usage: expseries x <number threads> <iterations>\n " );
    exit(-1);
  }
  else
  {
    sscanf(argv[1], "%lf", &x);
    sscanf(argv[2], "%d", &thread_count);
    sscanf(argv[3], "%d", &iterations);
  }

  printf("DBL_EPSILON = %le\n", DBL_EPSILON);

// The goal of a series computation is to avoid use of the math library completely
// and to use only basic arithmetic to compute transcendental and polynomial
// functions.
//
  for(idx=0; idx < iterations; idx++) 
  {
    num=x, fact=1.0, term=1.0;

    for(jdx=1; jdx <= idx; jdx++)
    {

        // The order of operations to avoid overflow in this series is very important
        // as factorial will go to inf fast, so we need a recurrence relation that
        // keeps numbers in range for 15 digits of precision and max value with the
        // largest exponent
        term *= x/((double)(jdx));

        // num and fact will eventually both go to inf
        num *= x;
        fact *= (double)(jdx);
    }

    sum += term; cnt++;

    err = fabs(sum - exp(x));

    if(err >= preverr) 
    {
        printf("DONE with err = %le\n", err);
        break;
    }

    preverr = err;

  }

  printf ( "C math library exp(x)  =%15.14lf\n", exp(x));
  printf ( "Series exp(x) =%15.14lf, iterations=%d, error=%le, num=%le,  fact=%le\n", sum, cnt, fabs(sum - exp(x)), num, fact);

}

