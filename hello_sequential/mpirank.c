#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <unistd.h>

int main(void)
{
    int comm_sz, my_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    printf("Worker %d: hello\n", my_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    if(my_rank == 0) printf("TEST COMPLETE\n");

    MPI_Finalize();
    return 0;
}

