#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

int main(void)
{
  int comm_sz, my_rank, idx;
  unsigned int local_sum=0, g_sum=0;

  //unsigned int ranksumd=0;
  //MPI_Status status;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank != 0) 
  {
    // sum the digits up to rank for each non 0 rank
    for(idx = 0; idx <= my_rank; idx++)
      local_sum += idx;

    // send sum to 0 rank process
    //MPI_Send(&local_sum, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
  }

#if 0
  else 
  {

    for(int q=1; q < comm_sz; q++) 
    {
      // Receive each rank sum
      MPI_Recv(&local_sum, 1, MPI_UNSIGNED, q, 0, MPI_COMM_WORLD, &status);
      ranksumd = status.MPI_SOURCE;

      // sum up the rank sums
      g_sum += local_sum;


      printf("sum=%u from %u expect=%u\n", 
             local_sum, status.MPI_SOURCE, (ranksumd*(ranksumd+1)/2));

    }

  }
#endif

  MPI_Reduce(&local_sum, &g_sum, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

  // collective comm broadcast the rank 0 g_sum to all other ranks
  MPI_Bcast(&g_sum, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);


   // all other ranks - sum of rank sums
  printf("my_rank=%d, sum of rank sums=%u\n", my_rank, g_sum);

  MPI_Finalize();

  return 0;
}
