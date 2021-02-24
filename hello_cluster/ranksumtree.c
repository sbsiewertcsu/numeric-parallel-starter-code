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

  // rank sum of digits
  if(my_rank != 0) 
  {
    // sum the digits up to rank for each non 0 rank
    for(idx = 0; idx <= my_rank; idx++)
      local_sum += idx;

    printf("sum=%u for %d expect=%u\n", local_sum, my_rank, (my_rank*(my_rank+1))/2);
  }

  g_sum +=local_sum;

  MPI_Barrier(MPI_COMM_WORLD);

  // Up to comm_sz-1 start the tree
  shift=1;

  // Tree summation
  while(shift < comm_sz)
  {
    if(my_rank % (shift*2) == 0) // even single, pair, quad, etc.
    {
      MPI_Recv(&local_sum, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      g_sum +=local_sum;
      printf("my_rank=%d received from %d, local_sum=%u, g_sum=%u\n", my_rank, status.MPI_SOURCE, local_sum, g_sum);
      local_sum=g_sum;
    }
    else if((shift == 1) || (my_rank % shift == 0))
    {
      // send left
      printf("my_rank=%d shift=%d, send to %d\n", my_rank, shift, my_rank-shift);
      MPI_Send(&local_sum, 1, MPI_UNSIGNED, my_rank-shift, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    shift = shift * 2;
  }  


  MPI_Barrier(MPI_COMM_WORLD);

  // Tree distribution
  printf("Start distribution for my_rank=%d\n", my_rank);


  MPI_Barrier(MPI_COMM_WORLD);

  printf("my_rank=%d, sum of rank sums=%u\n", my_rank, g_sum);

  MPI_Finalize();

  return 0;
}
