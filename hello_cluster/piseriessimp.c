#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

// The 2 series used for computation of pi are documented well by Wikipedia
//
// https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80
//
// The first series is simply 1 - 1/3 + 1/5 - 1/7 + 1/9 - ... = pi/4
//
//

int main(int argc, char** argv)
{
  int comm_sz, my_rank, idx;
  double local_sum=0.0, g_sum=0.0, local_num=1.0;
  unsigned int length;
  unsigned int sub_length;

  if(argc < 2)
  {
      printf("usage: piseriesreduce <series n>\n");
      exit(-1);
  }
  else
  {
      sscanf(argv[1], "%u", &length);
  }


  // MPI initializaiton
  //
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


  // Dividing Work - we use a sub-interval for the series computations and we'll sum the sums
  // to get the final result after all MPI workers are done.
  //
  // The comm_sz is the # of workers we will use to divide up the work.
  //
  sub_length = length / comm_sz;


  // sum the sub-series for the rank for Leibniz's formula for pi/4
  //
  // Note that the range is rank specific - each range is 1/n of the total computation,
  // so look carefully at the initial index value, loop termination value, and increment.
  //
  for(idx = my_rank*sub_length; idx < (sub_length*(my_rank+1)); idx++)
  {
    local_sum += local_num / ((2.0 * (double)idx) + 1.0);
    local_num = -local_num;
  }

  printf("my_rank=%d, iterated up to %d, local_sum=%15.14lf\n", my_rank, idx, local_sum);


  // Collect computations done over sub-intervals by all rank 0...n workers and sum the sums!
  //
  //  This is way simpler than loops with send from each rank and receive in rank 0 for all the
  //  sub-interval computations and is automatic - saves lots of coding.
  //
  MPI_Reduce(&local_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


  if(my_rank == 0)
  {
      printf("20 decimals of pi  =3.14159265358979323846\n");
      printf("C math library pi  =%15.14lf\n", M_PI);
      printf("Madhava-Leibniz pi =%15.14lf, ppb error=%15.14lf\n", (4.0*g_sum), 1000000000.0*(M_PI - (4.0*g_sum)));
      printf("Sin(3.14159265358979323846)=%lf, Sin(4.0*g_sum)=%lf\n", sin(3.14159265358979323846), sin(4.0*g_sum));
  }

  MPI_Finalize();

  return 0;
}
