#include <stdio.h>
#include <stdlib.h>

#define ITERATIONS (1024)


int main(int argc, char *argv[])
{
    int idx=0;
    double v0[ITERATIONS], v1[ITERATIONS], v2[ITERATIONS];

    for(idx=0; idx < ITERATIONS; idx++)
        v0[idx]=(double)idx, v1[idx]=(double)idx+1.0, v2[idx]=(double)idx+2.0;

//#pragma omp parallel for
#pragma omp simd
    for(idx=0; idx < ITERATIONS; idx++)
    {
        v0[idx] = v1[idx] + v2[idx];
    }

    for(idx=0; idx < ITERATIONS; idx++)
        printf("v0[%d]=%lf\n", idx, v0[idx]);

    return 0;
}
