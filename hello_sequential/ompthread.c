#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS (10)

void helloThread(void);

int main(void)
{

#pragma omp parallel num_threads(NUM_THREADS)
  helloThread();

  printf ("TEST COMPLETE\n");
}


void helloThread(void)
{
  //int thread_count = omp_get_num_threads();
  int worker = omp_get_thread_num();

  printf("Thread %d: hello\n", worker);

  return;
} 
