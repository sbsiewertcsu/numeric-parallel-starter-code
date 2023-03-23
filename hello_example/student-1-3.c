#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS (10)

void helloThread(int*);

int main(void)
{
int sum = 0;

#pragma omp parallel num_threads(NUM_THREADS)
helloThread(&sum);

//check hard coded for output purposes. this is the only test case we care about
printf("gsum[0...99]=%d, check=%d\n", sum, 4950);

printf("TEST COMPLETE\n");
}


void helloThread(int* sum)
{
//int thread_count = omp_get_num_threads();
int worker = omp_get_thread_num();

//start for each omp thread
int start = worker * 10;
//end of each omp thread is start + 9
int end = start + 9;
//initialize sum to be returned
int local_sum = 0;

for (int i = start; i <= end; i++) {
local_sum += i;
}

printf("Thread %d: Counter sum[%d...%d]=%d\n", worker, start, end, local_sum);

//use atomic to add local_sum to the global sum
#pragma omp atomic
*sum += local_sum;

//return the final sum (Should be 4950)
return;
}
