// This is the simplest brute force stepping method possible to find a root
// on an interval with a fixed step size equal to error tolerance.
//
// It simply detects a sign change and declares the root to be half way between
// the step before the sign change and the step after and quits.
//
// Inspired by questions from Section 02 CSCI 551, 2022
//
// Why not just search by making small steps for a zero crossing.  This definitely works, but
// does require small step sizes, potentially many steps to search a large interval, and error
// will simply be stepsize / 2.0
//
// Makes an intersting comparison to Newton, Bisection, and Regula Falsi, all which base stepping
// on an estimate of where the root is on an interval.
//
// Generally, Newton should converge fastest, followed by Regula Falsi, then Bisection, and the Bruteroot will
// get the root stepping in one direction looking for a simple sign change, but is the slowest.
//
// This code has an added outter do-while loop to find all roots on a larger interval.
//
// Modified to handle "osculating" real roots - e.g., a parabola that "kisses" the X axis (intersects), but does
// not cross the X axis with a sign change.
//
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>

// First model a function with desmos.com for example to "see" roots
// 
// https://www.desmos.com/calculator
//
double f(double x)
{
    // Fill in your function here using the math library

    //return x*log10(x) - 1.2;
    //return cos(x*x);
    //return cos(x);
    //return sin(x);
    //return sin(x*x);

    //return((x*x)+(2.0*x)+1);
    return((x*x)+(2.0*x)+2);

    // Note this osculating parabola will break brute-root as there is no
    // sign change to detect - can we fix this?
    // Detect zero within tolerance, which here is < step
    //return((x*x)+(2.0*x)+1.0);

    // As can be seen with Desmos, this example equation has roots near: -2.851, 4.758, and 7.792
    // return (-(x*x*x) + 9.7*(x*x) -1.3*x -105.7);
}

int main(void)
{
    unsigned long long itr, maxitr, total_itr=0;
    unsigned int rootfound=0;
    double max_error=0.0, fstart=0.0, fstop=0.0;
    double step=2.0*DBL_EPSILON, x0, x1, xfinal, sign_changed=0, start;
    struct timespec start_t, stop_t;


    printf("smallest step is 2x epsilon is %20.19lf", (double)(2.0* DBL_EPSILON));

    printf("\nEnter x0 start, xfinal end, step size and maximum iterations to search for each root on sub-interval of 1.0\n");
    scanf("%lf %lf %lf %lld", &x0, &xfinal, &step, &maxitr);
    start=x0;

    printf("\nWill step at %20.19lf from %lf to %lf to find each root, with %lld total search steps\n\n",
           step, x0, xfinal, (((unsigned)(xfinal-x0)+1)*maxitr));
    printf("Starting search:\n");
    clock_gettime(CLOCK_MONOTONIC, &start_t);
    fstart = (double)start_t.tv_sec  + (double)start_t.tv_nsec / 1000000000.0;


    // while there are potentially more roots on the search interval
    //
    // exit outter loop search for all roots if end of search interval reached or
    // we have executed max iterations
    //
    while((x0 <  xfinal) && (total_itr < (maxitr*maxitr)))
    {
        for (itr=1; (itr<=maxitr); itr++) // find next root on sub-interval
        {

            x1=x0+step;

            // Any difference in sign between x0 and x1 will result in a sign_changed
            // less than zero - a crossing has been detected
            sign_changed = f(x0) * f(x1);

            // X-axis Crossing ROOT case
            //
            if(sign_changed < 0.0)
            {
                printf("\n*** Sign change at %lld iterations: estimated root at %16.15lf, f[root]=%16.15lf\n",
                       total_itr+itr, (x1+x0)/2.0, f((x1+x0)/2.0));

                if(fabs(f((x1+x0)/2.0)) > max_error) max_error=fabs(f((x1+x0)/2.0));

                rootfound++;

                // We can exit on first root (zero crossing found) or keep looking
                break;
            }

            // Osculating ROOT case
            //
            else if( fabs(f((x1+x0)/2.0)) < step) // hit zero within tolerance - special case for osculating functions
            {
                printf("\n*** Osculating ZERO at %lld iterations: estimated DOUBLE root at %16.15lf, f[root]=%16.15lf\n",
                       total_itr+itr, (x1+x0)/2.0, f((x1+x0)/2.0));

                if(fabs(f((x1+x0)/2.0)) > max_error) max_error=fabs(f((x1+x0)/2.0));

                // Because by math rules, this must be a double root
                rootfound=rootfound+2;

                // We need to now step away from the ZERO osculating intersection
                while( fabs(f((x1+x0)/2.0)) < step )
                {
                    x0=x1;
                    x1=x0+step;
                }

                // We can exit on first root (zero within tolerance) or keep looking
                break;
            }

            // End of search interval hit
            if(x0 > xfinal) { 
                printf("Hit end of serach interval\n"); break; 
            }

            x0=x1; // advance the step interval
        }

        // Re-start search for next root one step beyond last one found
        total_itr += itr;

        // Indicate searching progress
        printf(".");
        fflush(stdout);

        x0=x1;
        itr=1;

    } // end while

    clock_gettime(CLOCK_MONOTONIC, &stop_t);
    fstop = (double)stop_t.tv_sec  + (double)stop_t.tv_nsec / 1000000000.0;
    printf("\n\nSearch Results (competed in %lf milliseconds):\n", (fstop-fstart)*1000.0);

    if(!rootfound)
    {
        printf("After %lld iterations: No solution (zero crossing) found on interval %lf to %lf\n", itr, start, x0);
    }
    else
    {
        printf("%d roots found on interval %lf to %lf, with max error=%lf in %lld iterations\n",
               rootfound, start, x0, max_error, total_itr);
    }

    return 1;
}
