#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include <unistd.h>

double riemann_sum(double a, double b, int n, double step_size)
{
double sum = 0.0;
double x = a;
for(int i=0; i<n; i++)
{
sum += sin(x)*step_size;
x += step_size;
}
return sum;
}

int main(void)
{
int comm_sz, my_rank;
double a = 0.0;
double b = M_PI;
double n = 3141592;
double total_sum = 0.0;
double local_sum;
double local_n, local_a, local_b;

MPI_Init(NULL, NULL);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

double step_size = (b-a)/n;
local_n = n/comm_sz;

local_a = a + my_rank*local_n*step_size;
local_b = local_a + local_n*step_size;

local_sum = riemann_sum(local_a, local_b, local_n, step_size);

MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if(my_rank == 0) printf("Riemann sum of sine from %f to %f with step size %lf is %f\n", a, b, step_size, total_sum);

MPI_Finalize();
return 0;
}
