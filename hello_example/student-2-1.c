#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include <unistd.h>

int sum_range(int start, int end, int my_rank)
{
int sum = 0;
for(int i=start; i<=end; i++)
{
sum += i;
}
printf("Worker %d: Counter sum [%d..%d] = %d]\n", my_rank, start, end, sum);
return sum;
}

int main(void)
{
int comm_sz, my_rank;
int local_start, local_end;
int local_sum = 0;
int total_sum = 0;


MPI_Init(NULL, NULL);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

local_start = my_rank*10;
local_end = local_start + 9;

local_sum = sum_range(local_start, local_end, my_rank);
MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if(my_rank == 0) printf("gsum[0..99]=%d, check=%d\n", total_sum, (0 + 99) * 100 / 2);

MPI_Finalize();
return 0;
}
