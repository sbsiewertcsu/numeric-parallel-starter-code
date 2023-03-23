// Solution to problem #4 using OpenMP function style
//
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS (4) // could be argument for worker scaling, but keep it simple here

double function_to_integrate(double time);
double riemann_interval(double start, double end, double step_size);

int main(void)
{
    //variables for time, step size, sum
    double start = 0.0;
    double end = M_PI;
    double step_size = 0.000001;
    double gsum = 0.0;

#pragma omp parallel num_threads(NUM_THREADS)
    gsum += riemann_interval(start, end, step_size); 

    printf("Riemann sum of sine from %lf to %16.15lf = %16.15lf\n", start, end, gsum);
}


double riemann_interval(double start, double end, double step_size)
{
    double time;
    double interval_sum = 0.0;

    int my_thread = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    int start_interval = (my_thread*(end-start)/thread_count)/step_size;
    int end_interval = ((my_thread+1)*(end-start)/thread_count)/step_size;

    for (int idx = start_interval; idx < end_interval; idx++) 
    {
        time = ((double)idx)*step_size;
        interval_sum += function_to_integrate(time)*step_size;
    }

    return interval_sum;
}


double function_to_integrate(double time)
{
    return sin(time);
}
