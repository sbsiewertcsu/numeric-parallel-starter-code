#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Hello_thread(void);

int main(int argc, char *argv[])
{
    int thread_count;

    if(argc < 2)
        printf("usage: hello_omp <number threads>\n");
    else
    {
        sscanf(argv[1], "%d", &thread_count);
    }

#pragma omp parallel num_threads(thread_count)
    Hello_thread();

    return 0;
}

void Hello_thread(void)
{
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    printf("Hello from OMP thread %d of %d\n", my_rank, thread_count);
}
