// Solution for problem 5
//
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

#define STEP_SIZE (0.000001)  // make this an argument for problem scaling, but keep simple here
#define REDUCE_ELEMENTS (1)
#define MANAGER_RANK (0)

int main(void) 
{
    int comm_sz, my_rank;
    double step=STEP_SIZE, time=0.0;
    double local_area=0.0;
    double total_area=0.0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int total_steps = (M_PI/step);
    int steps_per_worker = total_steps/comm_sz;

    int local_start = my_rank*steps_per_worker;
    int local_end = (my_rank+1)*steps_per_worker;

    for(int idx=local_start; idx<local_end; idx++) 
    {
        time = (double)idx*step;
        local_area += sin(time)*step;
    }

    MPI_Reduce(&local_area, &total_area, REDUCE_ELEMENTS, MPI_DOUBLE, MPI_SUM, MANAGER_RANK, MPI_COMM_WORLD);

    if (my_rank == 0) 
        printf("Riemann sum of sine from %f to %f = %f\n", 0.0, M_PI, total_area);

    MPI_Finalize();

    return 0;
}

