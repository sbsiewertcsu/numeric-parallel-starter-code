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
// The second series is the summation of 2 / ((4n+1)(4n+3)) for n=0 to infinity
//

int main(int argc, char** argv)
{
  int comm_sz, my_rank, idx;
  double euler_local_sum=0.0, euler_g_sum=0.0;
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


  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  sub_length = length / comm_sz;

  if(my_rank == 0)
  printf("comm_sz=%d, length=%u, sub_length=%u\n", comm_sz, length, sub_length);

  // sum the sub-series for the rank for Leibniz's formula for pi/4
  for(idx = my_rank*sub_length; idx < (sub_length*(my_rank+1)); idx++)
  {
    local_sum += local_num / ((2.0 * (double)idx) + 1.0);
    local_num = -local_num;
  }

  printf("my_rank=%d, iterated up to %d, local_sum=%15.14lf\n", my_rank, idx, local_sum);


  // sum the sub-series for the rank for Euler improved convergence of the Madhava-Leibniz's formula for pi/4
  for(idx = my_rank*sub_length; idx < (sub_length*(my_rank+1)); idx++)
  {
    euler_local_sum += 2.0 / (((4.0 * (double)idx) + 1.0) * (4.0 * (double)idx + 3.0));
  }

  printf("my_rank=%d, iterated up to %d, local_sum=%15.14lf\n", my_rank, idx, local_sum);



  MPI_Reduce(&euler_local_sum, &euler_g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // collective comm broadcast the rank 0 g_sum to all other ranks
  //MPI_Bcast(&g_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


  if(my_rank == 0)
  {
      printf("20 decimals of pi  =3.14159265358979323846\n");
      printf("C math library pi  =%15.14lf\n", M_PI);
      printf("Madhava-Leibniz pi =%15.14lf, ppb error=%15.14lf\n", (4.0*g_sum), 1000000000.0*(M_PI - (4.0*g_sum)));
      printf("Euler modified pi  =%15.14lf, ppb error=%15.14lf\n", (4.0*euler_g_sum), 1000000000.0*(M_PI - (4.0*euler_g_sum)));
  }

  MPI_Finalize();

  return 0;
}
