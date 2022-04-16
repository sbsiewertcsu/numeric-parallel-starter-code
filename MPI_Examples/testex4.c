#include <stdio.h>
#include <math.h>

double ex4_accel(double time);
double ex4_vel(double time);
double ex4_pos(double time);


// For help with printf format for numbers:
//
// https://alvinalexander.com/programming/printf-format-cheat-sheet/

int main(void)
{
    // Based on Symobolic Calculus definite integrals only, no numerical integration
    printf("TRAIN STARTING STATE:\n");
    printf("ex4_accel=\t%30.15lf\n", ex4_accel(0.0));
    printf("ex4_vel  =\t%30.15lf\n", ex4_vel(0.0));
    printf("ex4_pos  =\t%30.15lf\n", ex4_pos(0.0));

    printf("TRAIN MIDPOINT STATE:\n");
    printf("ex4_accel=\t%30.15lf\n", ex4_accel(900.0));
    printf("ex4_vel  =\t%30.15lf\n", ex4_vel(900.0));
    printf("ex4_pos  =\t%30.15lf\n", ex4_pos(900.0));

    printf("TRAIN FINAL STATE:\n");
    printf("ex4_accel=\t%30.15lf\n", ex4_accel(1800.0));
    printf("ex4_vel  =\t%30.15lf\n", ex4_vel(1800.0));
    printf("ex4_pos  =\t%30.15lf\n", ex4_pos(1800.0));
}

double ex4_accel(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that acceleration will peak to result in translation of 122,000.0 meters
    //static double ascale=0.2365893166123;
    static double ascale=0.236589076381454;

    return (sin(time/tscale)*ascale);
}


// determined based on known anti-derivative of ex4_accel function
double ex4_vel(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static double vscale=0.236589076381454*(1800.0/(2.0*M_PI));

    return ((-cos(time/tscale)+1)*vscale);
}


// determined based on known anti-derivative of ex4_vel function
double ex4_pos(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static double vscale=0.236589076381454*(1800.0/(2.0*M_PI));

    return ((-tscale*(sin(time/tscale))+time)*vscale);
}


