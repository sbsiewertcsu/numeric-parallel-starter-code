#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <unistd.h>

//#include "sine.h"
#include "trap.h"

const int MAX_STRING = 100;

int main(void)
{
        char greeting[MAX_STRING];
        char hostname[MAX_STRING];
        char nodename[MPI_MAX_PROCESSOR_NAME];
        int comm_sz;
        int my_rank, namelen;
        double local_sum=0.0, g_sum=0.0;
        int tablelen = sizeof(DefaultProfile)/sizeof(double);
        int subrange, residual;

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

                // Now sum up the error between the LUT function and math model for acceleration
                for(int idx = my_rank*subrange; idx < (my_rank*subrange)+subrange-1; idx++)
                {
                    local_sum += DefaultProfile[idx];
                }

                sprintf(greeting, "Sum of error for rank %d of %d on %s is %lf", my_rank, comm_sz, nodename, local_sum);
                MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        else
        {
                gethostname(hostname, MAX_STRING);

                MPI_Get_processor_name(nodename, &namelen);

                // Now sum up the error between the LUT function and math model for acceleration
                for(int idx = 0; idx < subrange-1; idx++)
                {
                    local_sum += DefaultProfile[idx];
                }


                printf("Hello from process %d of %d on %s error is %lf\n", my_rank, comm_sz, nodename, local_sum);

                for(int q=1; q < comm_sz; q++)
                {
                        MPI_Recv(greeting, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        printf("%s\n", greeting);
                }
        }

        MPI_Reduce(&local_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if(my_rank == 0) printf("Rank 0 g_sum = %lf\n", g_sum);

        MPI_Finalize();

        return 0;
}

