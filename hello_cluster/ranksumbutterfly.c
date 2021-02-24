#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

int main(void)
{
  int comm_sz, my_rank, idx, shift;
  unsigned int local_sum=0, g_sum=0;
  MPI_Status status;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Compute rank sum for all ranks to start
  if(my_rank != 0) 
  {
    // sum the digits up to rank for each non 0 rank - the sum of rank 0 is 0
    for(idx = 0; idx <= my_rank; idx++)
      local_sum += idx;

    printf("sum=%u for %d expect=%u\n", local_sum, my_rank, (my_rank*(my_rank+1))/2);
  }

  g_sum +=local_sum;

  MPI_Barrier(MPI_COMM_WORLD);

  // Up to comm_sz-1 start the butterfly
  //
  shift=1;

  while(shift < comm_sz)
  {

    if((my_rank/shift) % 2 == 0) // even single, pair, quad, etc.
    {
      // send shifting right
      //printf("my_rank=%d, shift=%d, send to %d\n", my_rank, shift, my_rank+shift);
      MPI_Send(&local_sum, 1, MPI_UNSIGNED, my_rank+shift, 0, MPI_COMM_WORLD);
    }
    else // odd single, pair, quad, etc.
    {
      // send shifting left
      //printf("my_rank=%d shift=%d, send to %d\n", my_rank, shift, my_rank-shift);
      MPI_Send(&local_sum, 1, MPI_UNSIGNED, my_rank-shift, 0, MPI_COMM_WORLD);
    }
  
    MPI_Barrier(MPI_COMM_WORLD);

    // receive from any rank that sent 
    MPI_Recv(&local_sum, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    g_sum +=local_sum;
    //printf("my_rank=%d received from %d, local_sum=%u, g_sum=%u\n", my_rank, status.MPI_SOURCE, local_sum, g_sum);
    local_sum=g_sum;

    MPI_Barrier(MPI_COMM_WORLD);

    shift = shift * 2;
  }

  printf("my_rank=%d, sum of rank sums=%u\n", my_rank, g_sum);

  MPI_Finalize();

  return 0;
}
