#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <unistd.h>

const int MAX_STRING = 100;

int main(void)
{
        char greeting[MAX_STRING];
        char hostname[MAX_STRING];
        char nodename[MPI_MAX_PROCESSOR_NAME];
        int comm_sz;
        int my_rank, namelen;

        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        if(my_rank != 0)
        {
                gethostname(hostname, MAX_STRING);

                MPI_Get_processor_name(nodename, &namelen);

                sprintf(greeting, "Hello from process %d of %d on %s\n", my_rank, comm_sz, nodename);
                MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        else
        {
                gethostname(hostname, MAX_STRING);

                MPI_Get_processor_name(nodename, &namelen);

                printf("Hello from process %d of %d on %s\n", my_rank, comm_sz, nodename);

                for(int q=1; q < comm_sz; q++)
                {
                        MPI_Recv(greeting, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        printf("%s\n", greeting);
                }
        }

        MPI_Finalize();
        return 0;
}

