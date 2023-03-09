#include <stdio.h>
#include <math.h>

double function_to_integrate(double time);
double RiemannSum(double start, double end, double step_size);


int main(void)
{
    printf("Riemann sum of sine from %lf to %16.15lf = %16.15lf\n", 0.0, M_PI, RiemannSum(0.0, M_PI, 0.000001));
}


double function_to_integrate(double time)
{
    return sin(time);
}


double RiemannSum(double start, double end, double step_size)
{
    int steps = (end - start)/step_size;
    double time, sum=0.0;

    for(int idx=0; idx < steps; idx++)
    {
        time = ((double)idx)*step_size;
        sum += function_to_integrate(time)*step_size;
    }

    return sum;
}

