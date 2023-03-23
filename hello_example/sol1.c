// Solution to problem #1
//
// Summing digits with MPI workers - divide work and complete, then merge results with reduce

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

#define SUM_DIGITS_RANGE (100)  // could become argument for problem scaling, but keep it simple here
#define REDUCE_ELEMENTS (1)
#define MANAGER_RANK (0)

int main(int argc, char* argv[]) 
{
    int comm_sz, my_rank, start, end;
    int local_sum = 0;
    int gsum = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    start = my_rank*(SUM_DIGITS_RANGE/comm_sz);
    end = (my_rank+1)*(SUM_DIGITS_RANGE/comm_sz);

    for(int idx=start; idx<end; idx++) 
    {
        local_sum += idx;
    }

    MPI_Reduce(&local_sum, &gsum, REDUCE_ELEMENTS, MPI_INT, MPI_SUM, MANAGER_RANK, MPI_COMM_WORLD);


    // Print overall sum and sum for each worker range
    for(int idx=0; idx < comm_sz; idx++)
    {
        if(my_rank == idx)
            printf("Worker %d: Counter sum[%d...%d]=%d\n", my_rank, start, end-1, local_sum);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if(my_rank == 0) 
        printf("gsum[0...99]=%d, check=%d\n", gsum, (SUM_DIGITS_RANGE*(SUM_DIGITS_RANGE-1)/2) );


    MPI_Finalize();

    return 0;
}
