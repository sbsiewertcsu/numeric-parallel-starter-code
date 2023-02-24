#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <unistd.h>

// Symmetric function tables used in Ex #3 and #4
#include "sine.h"
//#include "trap.h"

const int MAX_STRING = 100;

int main(void)
{
    char greeting[MAX_STRING];
    char hostname[MAX_STRING];
    char nodename[MPI_MAX_PROCESSOR_NAME];
    int comm_sz;
    int my_rank, namelen;

    // Parallel summing example
    double local_sum=0.0, g_sum=0.0;
    double defaults[sizeof(DefaultProfile)/sizeof(double)];
    int tablelen = sizeof(DefaultProfile)/sizeof(double);
    int subrange, residual;

    // Fill in diff array used to simulate a new table of values, such as
    // a velocity table derived by integrating acceleration
    //
    for(int idx = 0; idx < tablelen-1; idx++)
        defaults[idx]=10.0;

    defaults[tablelen-1]=0.0;


    printf("Will divide up work for input table of size = %d\n", tablelen);
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    subrange = tablelen / comm_sz;
    residual = tablelen % comm_sz;

    printf("Went parallel: rank %d of %d doing work %d with residual %d\n", my_rank, comm_sz, subrange, residual);

    if(my_rank != 0)
    {
        gethostname(hostname, MAX_STRING);

        MPI_Get_processor_name(nodename, &namelen);

        // Now sum up the differences between the LUT function and ZERO
        for(int idx = my_rank*subrange; idx < (my_rank*subrange)+subrange; idx++)
        {
            local_sum += DefaultProfile[idx];

            // Simply shows each rank has it's own subset of the data with a sub-range over-written
            // with orignial data
            defaults[idx] = 0.0;
        }

        sprintf(greeting, "Sum of differences for rank %d of %d on %s is %lf", my_rank, comm_sz, nodename, local_sum);
        MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        gethostname(hostname, MAX_STRING);

        MPI_Get_processor_name(nodename, &namelen);

        // Now sum up the differences between the LUT function and ZERO
        for(int idx = 0; idx < subrange; idx++)
        {
            local_sum += DefaultProfile[idx];

            // Simply shows each rank has it's own subset of the date
            defaults[idx] = 0.0;
        }


        printf("Hello from process %d of %d on %s differences is %lf\n", my_rank, comm_sz, nodename, local_sum);

        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(greeting, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", greeting);
        }
    }

    MPI_Reduce(&local_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(my_rank == 0) printf("Rank 0 g_sum = %lf\n", g_sum);

    local_sum=0;

    MPI_Barrier(MPI_COMM_WORLD);

    // Now, each rank should just have part of the new defaults table, so none should sum to zero
    for(int idx = 0; idx < tablelen; idx++)
        local_sum += defaults[idx];

    printf("Rank %d sum of defaults = %lf\n", my_rank, local_sum);


    // Now to correct and overwrite all defaults, update all ranks with full table by sending
    // portion of table from each rank > 0 to rank=0, to fill in missing default data

    local_sum=0;

    if(my_rank != 0)
    {
        MPI_Send(&defaults[my_rank*subrange], subrange, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(&defaults[q*subrange], subrange, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Make sure all ranks have the full new default table
    MPI_Bcast(&defaults[0], tablelen, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Now rank zero has the data from each of the other ranks in one new table
    for(int idx = 0; idx < tablelen; idx++)
        local_sum += defaults[idx];

    printf("Rank %d sum of defaults = %lf\n", my_rank, local_sum);

    MPI_Finalize();

    return 0;
}

