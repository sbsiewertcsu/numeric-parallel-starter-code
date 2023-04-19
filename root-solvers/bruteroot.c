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
#include <stdio.h>
#include <math.h>
#include <float.h>

// First model a function with desmos.com for example to "see" roots
// 
// https://www.desmos.com/calculator
//
double f(double x)
{
    // Fill in your function here using the math library

    //return x*log10(x) - 1.2;
    //return cos(x*x);

    // As can be seen with Desmos, this example equation has roots near: -2.851, 4.758, and 7.792
    return (-(x*x*x) + 9.7*(x*x) -1.3*x -105.7);
}

int main(void)
{
    unsigned int itr, maxitr, rootfound=0, total_itr=0;
    double max_error=0.0;
    double step=2.0*DBL_EPSILON, x0, x1, xfinal, sign_changed=0, start;

    printf("\nEnter x0 start, xfinal end, step size and maximum iterations to search for each root on sub-interval of 1.0\n");
    scanf("%lf %lf %lf %d", &x0, &xfinal, &step, &maxitr);
    start=x0;

    printf("\nWill step at %lf from %lf to %lf to find each root, with %d total search steps\n\n",
           step, x0, xfinal, (((unsigned)(xfinal-x0)+1)*maxitr));
    printf("Starting search:\n");

    do // while there are potentially more roots on the search interval
    {
        for (itr=1; (itr<=maxitr); itr++) // find next root on sub-interval
        {

            x1=x0+step;

            // Any difference in sign between x0 and x1 will result in a sign_changed less than zero - a crossing
            sign_changed = f(x0) * f(x1);

            if(sign_changed < 0.0)
            {
                printf("\n*** Sign change at %3d iterations: estimated root at %16.15lf, f[root]=%16.15lf\n",
                       total_itr+itr, (x1+x0)/2.0, f((x1+x0)/2.0));
                if(fabs(f((x1+x0)/2.0)) > max_error) max_error=fabs(f((x1+x0)/2.0));
                rootfound++;

                // We can exit on first root (zero crossing found) or keep looking
                break;
            }

            x0=x1; // advance the step interval
        }

        // Re-start search for next root one step beyond last one found
        total_itr += itr;

        // Indicate searching progress
        printf(".");
        //printf("Continuing serach after %d total iterations\n", total_itr);

        x0=x1;
        itr=1;

    } while((x0 <  xfinal) && (itr < maxitr)); // exit outter loop search for all roots if end reached or max iterations

    printf("\n\nSearch Results:\n");

    if(!rootfound)
    {
        printf("After %d iterations: No solution (zero crossing) found on interval %lf to %lf\n", itr, start, x0);
    }
    else
    {
        printf("%d roots found on interval %lf to %lf, with max error=%lf in %d iterations\n",
               rootfound, start, x0, max_error, total_itr);
    }

    return 1;
}
