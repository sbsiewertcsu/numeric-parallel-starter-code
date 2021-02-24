#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

int main(void)
{
  int comm_sz, my_rank, idx;
  unsigned int local_sum=0, g_sum=0;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank != 0) 
  {
    // sum the digits up to rank for each non 0 rank
    for(idx = 0; idx <= my_rank; idx++)
      local_sum += idx;

     printf("sum=%u for %u expect=%u\n",
             local_sum, my_rank, (my_rank*(my_rank+1)/2));

  }

  MPI_Allreduce(&local_sum, &g_sum, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

   // all ranks - sum of rank sums
  printf("my_rank=%d, sum of rank sums=%u\n", my_rank, g_sum);

  MPI_Finalize();

  return 0;
}
