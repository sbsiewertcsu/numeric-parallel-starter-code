#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>


unsigned long int counterThread(int start, int end)
{
    int i;
    unsigned long int sum=0;

    for(i=start; i < end; i++)
    {
        sum=sum+i;
    }

    //printf("sum[%d...%d]=%lu\n", start, end-1, sum);
    return sum;
}

#define MAX_RANGE (1000000000)
#define MAX_ITERATIONS (10)

int main (int argc, char *argv[])
{
   int numworkers=10, start, end, iter, idx;
   unsigned long int sum=0, check;

   if(argc == 2)
   {
       sscanf(argv[1], "%d", &numworkers);
       printf("will run with %d threads for %d iterations\n", numworkers, MAX_ITERATIONS);
   }
   else
   {
       printf("will run with default %d threads for %d iterations\n", numworkers, MAX_ITERATIONS);
   }

   if(((numworkers % 10) != 0) && (numworkers != 1))
   {
       printf("uneven division of summing range\n");
       exit(-1);
   }


#pragma omp parallel for private(iter, idx, start, end) reduction(max:sum) num_threads(numworkers)
   for(iter=0; iter < MAX_ITERATIONS; iter++)
   {
       sum = (unsigned long)MAX_RANGE;
       start=0;
       end=(MAX_RANGE / numworkers);

       for(idx=0; idx < numworkers; idx++)
       {
           sum += counterThread(start, end);

           start = end;
           end = end + (MAX_RANGE / numworkers);
       }
   }


   // main thread adds the last number in 0...MAX_RANGE
   //
   // all other threads sum ranges in 0...MAX_RANGE-1
   check = ((unsigned long)MAX_RANGE * ((unsigned long)MAX_RANGE+1))/2;

   printf("sum of sums = %lu, check = %lu\n", sum, check);

   printf("TEST COMPLETE\n");
}
