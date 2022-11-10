//https://www.codewithc.com/c-program-for-regula-falsi-method/
//
// Refactored to improve variable names and comments
// 
// Sam Siewert, 4/26/22
//
// Regula Falsi does not require the derivative of the function to estimate a new interval
// that contains the root like Newton-Raphson, but rather uses the SECANT, which can simply be
// determined by evaluating the function at each end of the search interval.
//
// The intersection of the SECANT based upon slope, will ALWAYS contain the ZERO either on the 
// LEFT side of the search interval or the RIGHT side.
//
// We simply pick the side that contains the ZERO and iterate.
//
// Theory can be understood by starting with Wikipedia - https://en.wikipedia.org/wiki/Regula_falsi
//
// The double false position is the fastest converging and most reliable.
//
#include<stdio.h>
#include<math.h>

double f(double x)
{
    //return cos(x) - x*exp(x);

    // As can be seen with Desmos, this example equation has roots near: -2.851, 4.758, and 7.792
    return (-(x*x*x) + 9.7*(x*x) -1.3*x -105.7);
    //return cos(x*x);
}

void regula (double *x, double x0, double x1, double fx0, double fx1, int *itr)
{

    // Comput new x based on secant method rather than just stepping brute force or
    // using the slope X axis intercept, which requires df/dx.
    //
    // slope of secant is rise/run, which is [f(x1) - f(x0)] / [x1 - x0]
    //
    // The new x that intercepts the X axis, has the SAME SLOPE AS THE SECANT.
    //
    // So, [f(x0) - 0] / [x0 - x] = [f(x1) - f(x0)] / [x1 - x0]
    //
    // Or, rearranging, we get [x1 - x0] * f(x0) / [x0 - x] = f(x1) - f(x0),
    //
    // which can be further rearranged to get:
    //
    // x0 - x = [x1 - x0] * f(x0) / [f(x1 - f(x0)]
    // x = x0 - [x1 - x0] / [f(x1) - f(x0)] * f(x0)
    //
    *x = x0 - ((x1 - x0) / (fx1 - fx0))*fx0;

    ++(*itr);

    printf("Iteration no. %3d X = %7.5f \n", *itr, *x);
}


int main(void)
{
    int itr = 0, maxmitr;
    double x0,x1,x2,x3,allerr;

    printf("\nEnter the values of x0, x1, allowed error and maximum iterations:\n");
    scanf("%lf %lf %lf %d", &x0, &x1, &allerr, &maxmitr);

    printf("x0=%lf, x1=%lf, err=%lf, iter=%d\n", x0, x1, allerr, maxmitr);


    // Get the first value for the intersection of the SECANT on the interval
    regula (&x2, x0, x1, f(x0), f(x1), &itr);

    do
    {
        // Recall that a ZERO crossing is anywhere where the value of the function at 2
        // different x values is negative.
        //
        // If we have a ZERO crossing between new x2 and x0, ZERO crossing between x1 and x2
        if (f(x0)*f(x2) < 0)
            x1=x2;

        // ELSE we have a ZERO crosssing between x0 and x2
        else
            x0=x2;


        // Get the new value for the intersection of the SECANT on the new interval
        regula (&x3, x0, x1, f(x0), f(x1), &itr);


        if (fabs(x3-x2) < allerr)
        {
            printf("After %d iterations, root = %20.15lf\n", itr, x3);
            return 0;
        }

        // Adjust the interval so it contains the ZERO between
        x2=x3;
    }
    while (itr<maxmitr);

    printf("Solution does not converge or iterations not sufficient:\n");
    return 1;
}
