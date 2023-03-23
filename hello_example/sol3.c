// Solution for problem 3 adapted from #2 pthreads
//
// Divide up work to sum digits with 10 threads and then sum the sums
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

#define NUM_THREADS 10          // could become argument for worker scaling, but keep it simple here
#define SUM_DIGITS_RANGE (100)  // could become argument for problem scaling, but keep it simple here

int gsum = 0;
int workerThread(void);


int main (int argc, char *argv[])
{
// thread print order may be different, but the problem allows this!
#pragma omp parallel num_threads(NUM_THREADS)
    gsum += workerThread();

    printf("gsum[0...99]=%d, check=%d\n", gsum, 4950);
    return 0;
}


int workerThread(void)
{
    int local_sum=0, start, end;
    int my_thread = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    start = my_thread*thread_count;
    end = (my_thread*thread_count)+(thread_count-1);

    for (int idx=start; idx<=end; idx++) 
        local_sum += idx;

    printf("Thread %d: Counter sum[%d...%d]=%d\n",
           my_thread, start, end, local_sum);

    return local_sum;
}
