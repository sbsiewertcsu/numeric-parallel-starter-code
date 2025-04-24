// https://www.codewithc.com/c-program-for-newton-raphson-method/
//
// Sam Siewert
//     * Refactored to improve variable names and commenting - 4/26/2022
//     * Improved exit condition to use "if (fabs(f(x1)) < allerr)", rather than h < tolerance
//
// CSCI 551 class suggestons
//     * Based on suggestion from fall 2022 class, I added a "nudge" feature to push bad guesses away
//       from a zero slope start.  Zero slope will prevent convergence. - 11/16/2022
//
// For theory overview, start with Wikipedia - https://en.wikipedia.org/wiki/Newton's_method
//
// Draw some examples and convince yourself that the slope of the tangent (derviative of the function) at x, 
// helps with the search for the ZERO crossing, which is a root of the function.
//
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

// model a function with desmos for example to "see" roots
// 
// https://www.desmos.com/calculator
//
// Make sure your derivative function df is in fact the derivative of f
//
double f(double x)
{

    // FUNCTION #1
    //
    // PARABOLA that touches X-axis (root), but does not cross it
    // where b^2 - 4ac =0, e.g., b=2, a=1, c=1 is a case like this.
    //
    //return((x*x)+2.0*x+1.0);

    // FUNCTION #2
    //return (-(x*x*x) + 9.7*(x*x) -1.3*x -105.7);

    // FUNCTION #3
    //return x*log10(x) - 1.2;

    // FUNCTION #4
    //return (sin(x*x));


    // FUNCTION #5
    //return (cos(x*x));

    // FUNCTION #6
    //return cos(x);

    // FUNCTION #7
    return sin(x);
}

// model the derivative of that function using calculus and check your answer
//
// https://www.derivative-calculator.net/
//
double df (double x)
{

    // FUNCTION #1 Slope
    //
    // Derivative of PARABOLA that touches X-axis (root), but does not cross it
    // where b^2 - 4ac =0, e.g., b=2, a=1, c=1 is a case like this.
    //
    //return((2.0*x)+2.0);

    // FUNCTION #2 Slope
    // As can be seen with Desmos, this example equation has roots near: -2.851, 4.758, and 7.792
    //return (-3.0*(x*x) + 19.4*(x) -1.3);

    // FUNCTION #3 Slope
    //return log10(x) + 0.43429;


    // FUNCTION #4 Slope
    //return (2.0*x*cos(x*x));

    // FUNCTION #5 Slope
    //return (-2.0*x*sin(x*x));

    // FUNCTION #6 Slope
    //return -sin(x);

    // FUNCTION #7 Slope
    return cos(x);

    // BAD SLOPES - can cause divergence and failure of Newton method
    //
    // Undefined slope, zero, or huge positive or negative slope can cause issues for Newton 
    // convergence as well as an INCORRECT derivative, although, sometimes an INCORRECT derivative will converge (more
    // slowly most often).
    //
    //return (0.0);
    //return (DBL_MAX);
    //return (-DBL_MAX);

}

int main(void)
{
    int itr, maxitr, nudge;
    double h, x0, x1, allerr, x0sav;

    printf("\nEnter x0, allowed error and maximum iterations\n");
    scanf("%lf %lf %d", &x0, &allerr, &maxitr);

    // GUESS ADJUSTMENT LOGIC - suggested by fall 2022 class
    //
    // Check for ZERO slope at guess - this will cause NR to diverge!
    //
    // A new and better guess will be required
    if(fabs(df(x0)) < allerr)
    {
        printf("Slope of function at guess is %lf, which is too close to zero\n", df(x0));

        x0 = x0 + allerr;

        for(nudge = 0; nudge < maxitr; nudge++)
        {
            if(fabs(df(x0)) < allerr)
                x0=x0+allerr;
            else
                break;
        }

        printf("Nudging guess %d times to %lf, and new slope is %lf\n", nudge, x0, df(x0));

        if(fabs(df(x0)) < allerr)
        {
            printf("Nudge did not work, still too close to zero after %d nudges... exiting\n", maxitr);
            printf("Try a new guess based on review of function on Desmos.com or INCREASE iterations\n");
            exit(-1);
        }
    }
    else
    {
        printf("Slope of function at guess is %lf\n", df(x0));
    }


    x0sav = x0;
    printf("x0=%lf, err=%lf, max iter=%d\n", x0, allerr, maxitr);

    for (itr=1; itr<=maxitr; itr++)
    {
        // Taking the slope at current guess of x0, we look for a step, h, where f(x1)=0
        //
        // Since SLOPE at x0, or df(x0) = [f(x0) - f(x1)] / [x0 - x1] or rise/run and f(x1)=0, where
        // the TANGENT SLOPE INTERSECTS the X axis, we know
        //
        // df(x0) = [f(x0) - 0] / [x0 - x1]
        //
        // df(x0) * [x0 - x1] = f(x0)
        //
        // Note that h is x0 - x1, the step
        //
        // So, h = f(x0) / df(x0)
        
        // This computation can result in "inf" for df(x0)=0.0, but we check for that before we start
        h=f(x0)/df(x0);

        // Compute next guess based on x0 based on h = x0 - x1
        x1=x0-h;

        printf(" At Iteration no. %3d, x = %9.6f, f[x1]=%15.14lf\n", itr, x1, f(x1));

        // Error is simply the absolut difference between x0 and x1 and with convergence, x0 and x1 eventually
        // become the same within our tolerance and we quit.
        //
        // This original logic seems flawed - it seems better to iterate until f(x1) < tolerance, not
        // just h < tolerance!
        //
        //if (fabs(h) < allerr) -- original test
        //
        if (fabs(f(x1)) < allerr)
        {
            printf("After %3d iterations, root = %20.15f\n", itr, x1);
            printf("CHECK: f[root]=%20.14lf, tolerance=%15.14lf, h=%15.14lf\n", f(x1), allerr, fabs(x0-x1));

            // Quit when we find a root that meets error tolerance
            return 0;
        }

        x0=x1;

    }

    // If we did not achieve our error tolerance before max iterations, indicate potential failure
    printf("x0=%lf, err=%lf, max iter=%d\n", x0sav, allerr, maxitr);
    printf(" The required solution does not converge or iterations are insufficient\n");
    return 1;
}
