#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

int main(void)
{
  int comm_sz, my_rank, idx;
  unsigned int local_mul=1, g_mul=1;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank != 0) 
  {
    // mul the digits up to rank for each non 0 rank
    for(idx = 0; idx < my_rank; idx++)
      local_mul = local_mul * (idx+1);

     printf("Factorial of my_rank=%d, mul=%u\n", my_rank, local_mul);

  }

  //MPI_Allreduce(&local_mul, &g_mul, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_mul, &g_mul, 1, MPI_UNSIGNED, MPI_PROD, MPI_COMM_WORLD);

   // What is this?
  printf("my_rank=%d, local_mul=%d, mul of local_muls=%u\n", my_rank, local_mul, g_mul);

  MPI_Finalize();

  return 0;
}
