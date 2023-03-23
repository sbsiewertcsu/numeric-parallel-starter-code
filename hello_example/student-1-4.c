
#include <stdio.h>
#include <math.h>
#include <omp.h>

double function_to_integrate(double time);
double RiemannSum(double start, double end, double step_size);

int main(void)
{
//variables for time, step size, sum
double start = 0.0;
double end = M_PI;
double step_size = 0.000001;
double sum = 0.0;
//pragma to create four threads, and uses reduction on sum variable for the final output
#pragma omp parallel num_threads(4) reduction(+:sum)
{
//current thread
int tid = omp_get_thread_num();

//total num threads, in order to find start and end values
int nthreads = omp_get_num_threads();

int start_of_section = tid * (end - start) / nthreads / step_size;
int end_of_section = (tid + 1) * (end - start) / nthreads / step_size;

double time;
double section_sum = 0.0;

for (int i = start_of_section; i < end_of_section; i++) {
//get time for this omp thread
time = ((double)i)*step_size;
//calc sum for this omp thread
section_sum += function_to_integrate(time)*step_size;
}
//calculate the final sum
sum += section_sum;
}
printf("Riemann sum of sine from %lf to %16.15lf = %16.15lf\n", start, end, sum);
}

double function_to_integrate(double time)
{
return sin(time);
}

double RiemannSum(double start, double end, double step_size)
{
int steps = (end - start) / step_size;
double time, sum = 0.0;
for (int idx = 0; idx < steps; idx++) {
time = ((double)idx)*step_size;
sum += function_to_integrate(time)*step_size;
}
return sum;
}

