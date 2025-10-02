#include <iostream>
#include <cmath>

using namespace std;

// Replace or pass your own function; keeping this here for the demo.
double sin_function(double x)
{
    return std::sin(x);
}

double const_function(double x)
{
    return 10.0;
}


// Integrate y'(x) = function_to_integrate(x), y(a)=0, using classic RK4.
// Returns y(b) = ∫_a^b function_to_integrate(x) dx.
//
// a  : lower bound of integration
// b  : upper bound of integration
// n  : number of steps (uniform)
// function_to_integrate : pointer to f(x)
double runge_kutta_integrate(double a, double b, int n,
                             double (*function_to_integrate)(double))
{
    const double h = (b - a) / static_cast<double>(n);

    // y approximates the integral value at the current x (y(a)=0)
    double y = 0.0;

    // March from a to b in n uniform steps
    for (int i = 0; i < n; i++)
    {
        const double x = a + i * h;

        // RK4 stages: k1..k4 evaluate f at x, x+h/2, x+h
        const double k1 = function_to_integrate(x);
        const double k2 = function_to_integrate(x + 0.5 * h);
        const double k3 = function_to_integrate(x + 0.5 * h);
        const double k4 = function_to_integrate(x + h);

        // Update the integral y using RK4 combination
        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    return y; // ≈ ∫_a^b f(x) dx
}

int main()
{
    const double a = 0.0;
    const double b = M_PI;
    const int    n = 1000000; // same scale as earlier examples

    const double result = runge_kutta_integrate(a, b, n, &sin_function);

    cout.setf(std::ios::fixed);
    cout.precision(15);
    cout << "The integral of sin(x) from " << a << " to " << b
         << " with " << n << " steps is " << result << endl;

    return 0;
}

