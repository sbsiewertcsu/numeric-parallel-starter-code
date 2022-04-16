#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double pi_subseries(int n);

int main(int argc, char *argv[])
{
  double sum = 0.0;
  int n, thread_count=2;

  if (argc < 3)
  {
    printf ( "usage: pi_omp <number threads> <iterations> \n " );
    exit(-1);
  }
  else
  {
    sscanf(argv[1], "%d", &thread_count);
    sscanf(argv[2], "%d", &n);
  }


// Make a parellel function rather than parallel loop
//
// Function must divide up work by thread # and then sum the sums, 
// which can be completed in any order as addition is fully associative.

#pragma omp parallel num_threads(thread_count)
  sum += pi_subseries(n);

  printf ( "20 decimals of pi  =3.14159265358979323846\n");
  printf ( "C math library pi  =%15.14lf \n " , M_PI);
  printf ( "Madhava-Leibniz pi =%15.14lf, ppb error=%15.14lf\n", (4.0*sum), 1000000000.0*(M_PI-(4.0*sum )));

}


// The series used for computation of pi is documented well by Wikipedia
//
// https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80
//
// The series is simply 1 - 1/3 + 1/5 - 1/7 + 1/9 - ... = pi/4
//

double pi_subseries(int n)
{
  // This is key to divide up work by thread
  int thread_count = omp_get_num_threads();
  int my_rank = omp_get_thread_num();
  int length = n / thread_count;
  int residual = n % thread_count;
  int iterations;

  int idx;
  double sum = 0.0;
  double num = 0.0;

  if(my_rank == (thread_count-1)) 
      iterations=length+residual;
  else
      iterations=length;

  printf("Thread %d of %d, length=%d, residual=%d, iterations=%d\n", my_rank, thread_count, length, residual, iterations);

  for (idx = my_rank*length; idx < (my_rank*length)+iterations; idx++) 
  {
      num=1.0/(1.0+2.0*idx);

      if (idx % 2 == 0)
         sum += num ;
      else
         sum -= num ;
  }

  return (sum);
} 
