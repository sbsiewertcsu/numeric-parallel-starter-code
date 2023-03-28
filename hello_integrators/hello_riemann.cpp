#include <iostream>
#include <cmath>

#define RANGE (M_PI)
#define STEPS (1000000)

using namespace std;

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

    for (int idx = 0; idx < n; idx++) 
    {
        double x = a + idx * h;
        double fx = function_to_integrate(x);

        // Add the value of the function at the left endpoint of each subinterval.
        sum += fx;
    }

    return h * sum;
}


int main() 
{
    double a = 0.0;
    double b = M_PI;
    int n = 1000000;

    double result = left_riemann_sum(a, b, n);

    cout.precision(15);
    cout << "The integral of f(x) from 0.0 to " << b << " with " << n << " steps is " << result << endl;

    return 0;
}

double function_to_integrate(double x)
{
    return (sin(x));
}
