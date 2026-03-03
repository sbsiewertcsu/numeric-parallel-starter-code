#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

int main(void)
{
  int comm_sz, my_rank, idx;
  unsigned int local_mul=1, rankmul=1;
  MPI_Status status;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank != 0) 
  {
    for(idx = 0; idx < my_rank; idx++)
      local_mul = local_mul * (idx+1);

    printf("my_rank is %d, local_mul=%d\n", my_rank, local_mul);
    MPI_Send(&local_mul, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
  }

  else 
  {
    for(int q=1; q < comm_sz; q++) 
    {
      //MPI_Recv(&local_mul, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&local_mul, 1, MPI_UNSIGNED, q, 0, MPI_COMM_WORLD, &status);
      rankmul = status.MPI_SOURCE;
      printf("local_mul=%u, rankmul=%u\n", local_mul, rankmul);
    }
  }

  MPI_Finalize();

  return 0;
}
