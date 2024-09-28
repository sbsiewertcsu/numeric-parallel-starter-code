#include <iostream>
#include <cmath>

#define RANGE (M_PI)
#define STEPS (1000000)

using namespace std;

int thread_count=1;

double function_to_integrate(double x);

////////////////////////////////////////////////////////////////////////////////
// Computes the definite integral of a given function using Left Riemann sum. //
//                                                                            //
// @param a         The lower bound of integration.                           //
// @param b         The upper bound of integration.                           //
// @param n         The number of steps to use in the approximation.          //
//                                                                            //
// @return          The approximate value of the definite integral.           //
////////////////////////////////////////////////////////////////////////////////
double left_riemann_sum(double a, double b, int n) 
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel for num_threads(thread_count) reduction(+:sum)
    for (int idx = 0; idx < n; idx++) 
    {
        double x = a + idx * h;
        double fx = function_to_integrate(x);

        // Add the value of the function at the left endpoint of each subinterval.
        sum += fx;
    }

    return h * sum;
}


double riemann_sum(double start, double end, int nstep) 
{
    double stepSize = (end - start) / nstep;
    double sum = 0.0;
    double x=0.0, fx=0.0;

#pragma omp parallel for num_threads(thread_count) reduction(+:sum)
    for (int idx = 0; idx < nstep; idx++) 
    {
        x = start + (idx * stepSize);
        fx = function_to_integrate(x);

        // Add the rectangle at the left endpoint of each subinterval.
        sum += fx*stepSize;
    }

    return sum;
}


int main(int argc, char* argv[]) 
{
    double a = 0.0;
    double b = RANGE;
    int n = 1000000;

    if(argc == 2)
    {
        sscanf(argv[1], "%d", &thread_count);
        printf("Will run with: thread_count=%d\n", thread_count);
    }
    else
    {
        printf("Will run with default: thread_count=%d\n", thread_count);
    }

    double result = left_riemann_sum(a, b, n);

    cout.precision(15);
    cout << "The integral of f(x) from 0.0 to " << b << " with " << n << " steps is " << result << endl;

    return 0;
}

double function_to_integrate(double x)
{
    return (sin(x));
}
