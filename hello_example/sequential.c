#include <stdlib.h>
#include <stdio.h>

#define NUM_COUNTERS (10)
#define STEP (10)

int counterSeq(int worker, int start, int end)
{
    int sum=0, idx;

    for(idx=start; idx < end+1; idx++)
        sum=sum+idx;

    printf("Worker %d: Counter sum[%d...%d]=%d\n", worker, start, end, sum);

    return sum;
}


int main (int argc, char *argv[])
{
   int worker=0, idx;
   int gsum=0;

   for(idx=0; idx < (NUM_COUNTERS*STEP); idx+=STEP)
   {
      gsum += counterSeq(worker, idx, idx+(STEP-1)); worker++;
   }

   printf("gsum[0...%d]=%d, check=%d\n", idx-1, gsum, (idx-1)*(idx)/2);
}
