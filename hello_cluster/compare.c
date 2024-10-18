// Simple MPI program to emulate what the Excel spreadsheet model does for:
//
// velocity(t) = Sum[(accel(t)]
//
// position(t) = Sum[velocity(t)]
//
// Using as many ranks as you wish - should ideally divide into 1800 to avoid possible residual errors.
//
// Due to loop carried dependencies, this requires basic dynamic programming to use the final condition of one
// rank as the initial condition of others at a later time, but adjusted after the computations are done in parallel!
//
// This is identical to what's done in the OpenMP version of this code, but information is shared via message passing rather
// than shared memory.
//
// Note: this program will be helpful for the Ex #4 train problem where the anti-derivatives are unknown and all functions must
//       be integrated numerically.
//
// Sam Siewert, 2/24/2023, Cal State Chico - https://sites.google.com/mail.csuchico.edu/sbsiewert/home
//
// Please use as you see fit, but please do cite and retain a link to this original source
// here or my github (https://github.com/sbsiewertcsu/numeric-parallel-starter-code)
//

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

#define DEBUG_TRACE

// Function tables used in Ex #3 and #4 as well as test profiles for sine and a constant
// All funcitons have 1801 entries for time = 0.0 to 1800.0
#include "ex4.h"
//#include "ex3.h"
//#include "sine.h"
//#include "const.h"

const int MAX_STRING = 512;

int main(void)
{
    char resultmsg[MAX_STRING];
    char hostname[MAX_STRING];
    char nodename[MAX_STRING];
    //char nodename[MPI_MAX_PROCESSOR_NAME];
    int comm_sz;
    int my_rank, namelen;

    // Parallel summing example
    double local_sum=0.0, g_sum=0.0;
    double default_sum[sizeof(DefaultProfile)/sizeof(double)];
    double default_sum_of_sums[sizeof(DefaultProfile)/sizeof(double)];
    int tablelen = sizeof(DefaultProfile)/sizeof(double);
    int subrange, residual;

    // Fill in default_sum array used to simulate a new table of values, such as
    // a velocity table derived by integrating acceleration
    //
    for(int idx = 0; idx < tablelen; idx++)
    {
        default_sum[idx]=0.0;
        default_sum_of_sums[idx]=0.0;
    }


    printf("Will divide up work for input table of size = %d\n", tablelen);
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    subrange = tablelen / comm_sz;
    residual = tablelen % comm_sz;

    printf("Went parallel: rank %d of %d doing work %d with residual %d\n", my_rank, comm_sz, subrange, residual);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // START PARALLEL PHASE 1: Sum original DefaultProfile LUT by rank
    //                         which is integration of accel(t) here.
    //
    if(my_rank != 0)
    {
        gethostname(hostname, MAX_STRING);
        MPI_Get_processor_name(nodename, &namelen);

        // Now sum up the values in the LUT function
        for(int idx = my_rank*subrange; idx < (my_rank*subrange)+subrange; idx++)
        {
            local_sum += DefaultProfile[idx];
            default_sum[idx] = local_sum; // Each rank has it's own subset of the data
        }

        sprintf(resultmsg, "Sum of DefaultProfile for rank %d of %d on %s is %lf", my_rank, comm_sz, nodename, local_sum);
        MPI_Send(resultmsg, strlen(resultmsg)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        gethostname(hostname, MAX_STRING);
        MPI_Get_processor_name(nodename, &namelen);

        // Now sum up the values in the LUT function
        for(int idx = 0; idx < subrange; idx++)
        {
            local_sum += DefaultProfile[idx];
            default_sum[idx] = local_sum; // Each rank has it's own subset of the data
        }

        printf("Sum of DefaultProfile for rank %d of %d on %s is %lf\n", my_rank, comm_sz, nodename, local_sum);

        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(resultmsg, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", resultmsg);
        }
    }

    // This should be the summation of DefaultProfile, which should match the spreadsheet for a train profile for dt=1.0
    MPI_Reduce(&local_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(my_rank == 0) printf("\nRank 0 g_sum = %lf\n", g_sum);

    MPI_Barrier(MPI_COMM_WORLD);

    // DISTRIBUTE RESULTS: Now to correct and overwrite all default_sum, update all ranks with full table by sending
    // portion of table from each rank > 0 to rank=0, to fill in missing default data
    if(my_rank != 0)
    {
        MPI_Send(&default_sum[my_rank*subrange], subrange, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(&default_sum[q*subrange], subrange, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Adjust so initial condition is ending conditon of prior sum - SUPER IMPORTANT to adjust initial condition offset
            for(int idx = q*subrange; idx < (q*subrange)+subrange; idx++)
                default_sum[idx] += default_sum[((q-1)*subrange)+subrange-1];
        }
    }
    // Make sure all ranks have the full new default table - overkill, so optimize by sending just what they need
    MPI_Bcast(&default_sum[0], tablelen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    //
    // END PARALLEL PHASE 1: Every rank has the same updated default_sum table now
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // TRACE: Just a double check on the FIRST MPI_Reduce and trace output of first phase
    if(my_rank == 0)
    {
#ifdef DEBUG_TRACE
        // Now rank zero has the data from each of the other ranks in one new table
        printf("\nTRACE: Rank %d sum of default_sum = %lf\n", my_rank, g_sum);
        for(int idx = 0; idx < tablelen; idx+=100)
            printf("t=%d: a=%lf for v=%lf\n", idx, DefaultProfile[idx-1], default_sum[idx-1]);
#endif
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // START PARALLEL PHASE 2: Now that all ranks have the new default_sum table, we can proceed to sum all of those sums as before
    //
    // Do the next round of summing from the new table, which is velocity(t) here, at same resolution
    // as the original accel(t) table with 1801 data points for time=0, ..., 1800.
    //
    local_sum=0;

    if(my_rank != 0)
    {
        // Now sum up the values in the new LUT function default_sum
        for(int idx = my_rank*subrange; idx < (my_rank*subrange)+subrange; idx++)
        {
            local_sum += default_sum[idx];
            default_sum_of_sums[idx] = local_sum; // Each rank has it's own subset of the data
        }
    }
    else
    {
        // Now sum up the values in the new LUT function default_sum
        for(int idx = 0; idx < subrange; idx++)
        {
            local_sum += default_sum[idx];
            default_sum_of_sums[idx] = local_sum; // Each rank has it's own subset of the data
        }
    }

    // This should be the summation of the sums, which should match the spreadsheet for a train profile for dt=1.0
    MPI_Reduce(&local_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(my_rank == 0) printf("\nRank 0 g_sum = %lf\n", g_sum);

    // DISTRIBUTE: Finally correct and overwrite all default_sum_of_sums, update all ranks with full table by sending
    // portion of table from each rank > 0 to rank=0, to fill in missing default data
    if(my_rank != 0)
    {
        MPI_Send(&default_sum_of_sums[my_rank*subrange], subrange, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(&default_sum_of_sums[q*subrange], subrange, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Adjust so initial condition is ending conditon of prior sum - SUPER IMPORTANT to adjust initial condition offset
            for(int idx = q*subrange; idx < (q*subrange)+subrange; idx++)
                default_sum_of_sums[idx] += default_sum_of_sums[((q-1)*subrange)+subrange-1];
        }
    }
    // Make sure all ranks have the full new default table - overkill, so optimize by sending just what they need
    MPI_Bcast(&default_sum[0], tablelen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    //
    // END PHASE 2: Every rank has the same updated default_sum_of_sums table now
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // TRACE: Final double check on the SECOND MPI_Reduce and trace output of second phase
    if(my_rank == 0)
    {
#ifdef DEBUG_TRACE
        // Now rank zero has the data from each of the other ranks in one new table
        printf("\nTRACE: Rank %d sum of default_sum = %lf\n", my_rank, g_sum);
        for(int idx = 0; idx < tablelen; idx+=100)
            printf("t=%d: a=%lf for v=%lf and p=%lf\n", idx, DefaultProfile[idx-1], default_sum[idx-1], default_sum_of_sums[idx-1]);
#endif
    }

    MPI_Finalize();
    return 0;
}

