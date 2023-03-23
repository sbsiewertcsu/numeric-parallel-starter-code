
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS (10)

int helloThread(void);

int main(void)
{
int sum = 0;
#pragma omp parallel num_threads(NUM_THREADS) reduction(+:sum)
sum = helloThread();

printf("gsum[0..99]=%d, check=%d\n", sum, (0 + 99) * 100 / 2);
}


int helloThread(void)
{
int worker = omp_get_thread_num();
int start = worker * 10;
int end = start + 9;
int sum = 0;
for (int i = start; i <= end; i++)
{
sum += i;
}
printf("Thread %d: Counter sum [%d..%d] = %d]\n", worker, start, end, sum);

return sum;
}

