#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int comm_sz, my_rank, idx;
  unsigned int length=0;
  unsigned int* numlist = NULL;
  int local_length;
  unsigned int*  local_numlist;

  if(argc < 2) { printf("usage: ranksumscattergather <list length>\n"); exit(-1); }
  else {
      sscanf(argv[1], "%u", &length);
      local_numlist = malloc(length * sizeof(unsigned int));
      bzero(local_numlist, length * sizeof(unsigned int));
  }

  MPI_Init(NULL, NULL); MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(comm_sz <= length)
    local_length = length / comm_sz;
  else { printf("list length must be >= the comm_sz\n"); MPI_Finalize(); return(-1); }

  if(my_rank == 0) {
      printf("number list length=%u\n", length);
      numlist = malloc(length * sizeof(unsigned int));
      printf("Enter list of unsigned numbers:\n");
      for(idx=0; idx < length; idx++) scanf("%u", &numlist[idx]);

      MPI_Scatter(numlist, local_length, MPI_UNSIGNED, local_numlist, local_length, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      free(numlist);
  }
  else { MPI_Scatter(numlist, local_length, MPI_UNSIGNED, local_numlist, local_length, MPI_UNSIGNED, 0, MPI_COMM_WORLD); }

  for(idx=0; idx < local_length; idx++) printf("my_rank=%d, list element=%u\n", my_rank, local_numlist[idx]);

  MPI_Barrier(MPI_COMM_WORLD);

  if(my_rank == 0) {
      numlist = malloc(length * sizeof(unsigned int));
      MPI_Gather(local_numlist, local_length, MPI_UNSIGNED, numlist, local_length, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

      for(idx=0; idx < length; idx++) printf("%u ", numlist[idx]);

      printf("\n");
      free(numlist);
  }
  else { MPI_Gather(local_numlist, local_length, MPI_UNSIGNED, numlist, local_length, MPI_UNSIGNED, 0, MPI_COMM_WORLD); }

  MPI_Finalize();
  return 0;
}
