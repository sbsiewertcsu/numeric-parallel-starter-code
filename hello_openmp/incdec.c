#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define COUNT  10

// Potentially Unsafe global
int gsum=0;

long int gsumreduce=0;

long int incThread(void)
{
    int i;
    int thread_count = omp_get_num_threads();
    int my_rank = omp_get_thread_num();
    long int lsum=0;


    for(i=0; i<COUNT; i++)
    {
#pragma omp critical
        gsum=gsum+i;

        lsum=lsum+i;

        printf("Increment %d for thread idx=%d of %d, lsum=%ld, gsum=%d\n", i, my_rank, thread_count, lsum, gsum);
    }

    return lsum;
}


long int decThread(void)
{
    int i;
    int thread_count = omp_get_num_threads();
    int my_rank = omp_get_thread_num();
    long int lsum=0;

    for(i=0; i<COUNT; i++)
    {
#pragma omp critical
        gsum=gsum-i;

        lsum=lsum-i;

        printf("Decrement %d for thread idx=%d of %d, lsum=%ld, gsum=%d\n", i, my_rank, thread_count, lsum, gsum);
    }

    return lsum;
}



int main (int argc, char *argv[])
{
   int thread_count=8;
   long int suminc=0, sumdec=0;

#pragma omp parallel num_threads(thread_count) reduction(+: gsumreduce)
   {
      suminc += incThread();
      sumdec += decThread();
      gsumreduce += suminc + sumdec;
   }
  
   printf("FINAL: gsum=%d, gsumreduce=%ld\n", gsum, gsumreduce);
   printf("TEST COMPLETE\n");
}
