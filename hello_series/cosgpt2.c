// Another ChatGPT program to compute cos(x) over a range comparing to the 
// math library.
//
// Sam Siewert - cleaned up, added RMSE and argv for number of iterations.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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


#define TS_ITERATIONS (1000000)

int main(int argc, char *argv[]) 
{
    int num_terms = TS_ITERATIONS;
    double step = 0.1;
    double max_angle = 8 * 3.14159265358979323846;

    double sum_squared_diff = 0;
    int num_points = 0;

    if(argc == 1)
    {
        printf("Using default %d iterations\n", TS_ITERATIONS);
    }
    if(argc == 2)
    {
        num_terms = atoi(argv[1]); 
        printf("Using default %d iterations\n", num_terms);
    }
    else
    {
        printf("Use: cosgpt [num_terms]\n");
    }

    printf("Angle (radians)\t\tTaylor Series\t\tActual Value\t\tDifference\n");

    for (double x = 0.0; x <= max_angle; x += step) 
    {
        double cos_approx = cos_taylor_series(x, num_terms);
        double cos_actual = cos(x);
        double difference = cos_approx - cos_actual;

        sum_squared_diff += difference * difference;
        num_points++;

        printf("%16.15lf\t%16.15lf\t%16.15lf\t%16.15lf\n", x, cos_approx, cos_actual, difference);
    }

    double mean_squared_error = sum_squared_diff / num_points;
    double root_mean_squared_error = sqrt(mean_squared_error);

    printf("Root Mean Squared Error: %20.15lf\n", root_mean_squared_error);

    return 0;
}

