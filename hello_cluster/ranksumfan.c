#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

int main(void)
{
  int comm_sz, my_rank, idx;
  unsigned int local_sum=0, ranksumd=0, g_sum=0;
  MPI_Status status;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // rank sum of digits
  if(my_rank != 0) 
  {
    // sum the digits up to rank for each non 0 rank
    for(idx = 0; idx <= my_rank; idx++)
      local_sum += idx;

    MPI_Send(&local_sum, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
  }
  else 
  {

    for(int q=1; q < comm_sz; q++) 
    {
      MPI_Recv(&local_sum, 1, MPI_UNSIGNED, q, 0, MPI_COMM_WORLD, &status);
      ranksumd = status.MPI_SOURCE;

      g_sum += local_sum;

      printf("sum=%u for %u expect=%u\n", local_sum, status.MPI_SOURCE, (ranksumd*(ranksumd+1)/2));
    }
  }

  if(my_rank == 0) 
  {

    for(int q=comm_sz-1; q > 0; q--) 
    {
      MPI_Send(&g_sum, 1, MPI_UNSIGNED, q, 0, MPI_COMM_WORLD);
    }
  }
  else 
  {
    MPI_Recv(&g_sum, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
  }

  printf("my_rank=%d, sum of rank sums=%u\n", my_rank, g_sum);

  MPI_Finalize();

  return 0;
}
