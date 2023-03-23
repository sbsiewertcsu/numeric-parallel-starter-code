#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>

#define STEP_SIZE 0.000001

int main(void) {
int comm_sz, my_rank;

MPI_Init(NULL, NULL);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

double local_start = my_rank * (M_PI / comm_sz);
double local_end = (my_rank + 1) * (M_PI / comm_sz);
double local_area = 0.0;
for (double i = local_start; i < local_end; i += STEP_SIZE) {
local_area += sin(i) * STEP_SIZE;
}

double result;
MPI_Reduce(&local_area, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if (my_rank == 0) {
printf("Riemann sum of sine from %f to %f = %f\n", 0.0, M_PI, result);
}

MPI_Finalize();
return 0;
}

