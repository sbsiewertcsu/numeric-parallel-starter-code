// This code was written by ChatGPT 4.0
//
// Prompted to not use C math library at all.
//
// Sam Siewert - cleaned up and tested
//
#include <stdio.h>

// Function to compute x raised to the power of n
double power(double x, int n) 
{
    double result = 1;
    for (int i = 0; i < n; i++) 
    {
        result *= x;
    }
    return result;
}

// Function to compute cos(x) using Taylor series
double cos_taylor_series(double x, int num_terms) 
{
    // Normalize the angle to the range of -π to π
    double pi = 3.14159265358979323846;

    while (x > pi) 
    {
        x -= 2 * pi;
    }
    while (x < -pi) 
    {
        x += 2 * pi;
    }

    double cos_x = 0;
    double term = 1;

    for (int n = 0; n < num_terms; n++) 
    {
        if (n > 0) 
        {
            term *= -x * x / ((2 * n) * (2 * n - 1));
        }
        cos_x += term;
    }
    return cos_x;
}


int main() 
{
    double x;
    int num_terms;

    printf("Enter the angle in radians: ");
    scanf("%lf", &x);

    printf("Enter the number of terms in the Taylor series: ");
    scanf("%d", &num_terms);

    double cos_approx = cos_taylor_series(x, num_terms);
    printf("cos(%lf) using Taylor series with %d terms: %lf\n", x, num_terms, cos_approx);

    return 0;
}

