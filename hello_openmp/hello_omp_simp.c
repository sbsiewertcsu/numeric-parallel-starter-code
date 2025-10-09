#include <stdio.h>
#include <stdlib.h>

void Hello_thread(void);

int main(int argc, char *argv[])
{
    int thread_count=1;

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
    printf("Hello from OMP thread\n");
}
