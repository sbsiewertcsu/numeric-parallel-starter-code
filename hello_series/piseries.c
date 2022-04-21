// The series used for computation of pi is documented well by Wikipedia
//
// https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80
//
// The series is simply 1 - 1/3 + 1/5 - 1/7 + 1/9 - ... = pi/4
//

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int argc, char* argv[])
{
  int thread_count;
  int iterations;
  double sum=0;

  if (argc < 3)
  {
    printf ( "usage: pi_omp <number threads> <iterations> \n " );
    exit(-1);
  }
  else
  {
    sscanf(argv[1], "%d", &thread_count);
    sscanf(argv[2], "%d", &iterations);
  }

// Create threads up to thread_count for the for loop, which will be un-rolled and divided, with sum reduction
// for each loop sub-series to sum the entire series in the loop.
#pragma omp parallel for reduction(+:sum) num_threads(thread_count)
  for(int idx=0; idx < iterations; idx++) 
  {
    double num=1.0/(1.0+2.0*idx);
    if (idx % 2 == 0)
      sum += num ;
    else
      sum -= num ;
  }

  printf ( "20 decimals of pi  =3.14159265358979323846\n");
  printf ( "C math library pi  =%15.14lf \n " , M_PI);
  printf ( "Madhava-Leibniz pi =%15.14lf, ppb error=%15.14lf\n", (4.0*sum), 1000000000.0*(M_PI-(4.0*sum )));

}

