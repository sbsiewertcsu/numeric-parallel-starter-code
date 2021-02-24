#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

int main(void)
{
  int comm_sz, my_rank, idx;
  unsigned int local_sum=0, ranksumd=0;
  MPI_Status status;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank != 0) 
  {
    for(idx = 0; idx <= my_rank; idx++)
      local_sum += idx;

    MPI_Send(&local_sum, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
  }

  else 
  {
    for(int q=1; q < comm_sz; q++) 
    {
      //MPI_Recv(&local_sum, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&local_sum, 1, MPI_UNSIGNED, q, 0, MPI_COMM_WORLD, &status);
      ranksumd = status.MPI_SOURCE;
      printf("sum=%u from %u expect=%u\n", local_sum, status.MPI_SOURCE, (ranksumd*(ranksumd+1)/2));
    }
  }

  MPI_Finalize();

  return 0;
}
