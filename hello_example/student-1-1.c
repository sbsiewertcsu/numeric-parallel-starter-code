#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <unistd.h>

int main(int argc, char* argv[]) {
int comm_sz, my_rank;
int localSum = 0;
int globalSum = 0;
int highestValue = 100;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

int start = my_rank * (highestValue / comm_sz);
int end = (my_rank + 1) * (highestValue / comm_sz);
for (int i = start; i < end; i++) {
localSum += i;
}

MPI_Reduce(&localSum, &globalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

if(my_rank == 0) {
printf("gsum[0...99]=%d, check=%d\n", globalSum, highestValue * (highestValue - 1) / 2);
}
else {
printf("Worker %d: Counter sum[%d...%d]=%d\n", my_rank, start, end - 1, localSum);
}

MPI_Finalize();
return 0;
}



