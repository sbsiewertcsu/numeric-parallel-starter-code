#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// For values between 1 second indexed data, use linear interpolation to determine profile value at any "t".
//
// Wikipedia - https://en.wikipedia.org/wiki/Linear_interpolation
//
// This assumes that the profile is "piecewise linear", which is accurate for linear or constant functions, and 
// reasonably accurate for most non-linear functions over small intervals of 1 second.
//
// This test program simply interpolates values between 1 second intervals to produce a value every 1/10th of 
// a second.

//#include "ex3accel.h"
#include "ex6lin.h"

// table look-up for acceleration profile given and velocity profile determined
//
// Note: for 2 functions (2 trains) we would want to make 2 different versions of this
//       function or better yet, pass in the table to use.
//
double table_accel(int timeidx);


// indirect generation of acceleration or velocity at any time with table interpolation
//
// Note: for 2 functions (2 trains) we would want to make 2 different versions of this
//       function that each uses the correct table.
double faccel(double time);

// This is the number of interpolation steps (data to generate) between table data values
#define STEPS_PER_SEC (10)

void main(int argc, char *argv[])
{
    int idx;
    int steps_per_sec=STEPS_PER_SEC;
    double time, dt;


    printf("argc=%d, argv[0]=%s\n", argc, argv[0]);

    if(argc == 2)
    {
        sscanf(argv[1], "%d", &steps_per_sec);
        printf("Will use %d steps per sec\n", steps_per_sec);
    }
    else
    {
        printf("Use: trainprofile [steps_per_sec]\n");
        printf("Will use default time profile\n");
    }

    dt = 1.0 / steps_per_sec;

    printf("Step size of %lf\n", dt);

    // Verify the static initializer or loaded CSV array for size and values
    //
    // And use it to generate values between 1 second given values.
    //
    printf("Number of values in profile = %lu for 1801 expected\n", sizeof(DefaultProfile)/sizeof(double));

    for(idx=0; idx <= (STEPS_PER_SEC*1800); idx++)
    {
        // time you would use in your integrator and faccel(time) is the fuction to integrate
        time = 0.0 + (dt*(double)idx);
        printf("%015.14lf, %015.14lf\n", time, faccel(time));
    }

}


// Simple look-up in accleration profile array
//
// Added array bounds check for known size of train arrays
//
double table_accel(int timeidx)
{
    long unsigned int tsize = sizeof(DefaultProfile) / sizeof(double);

    // Check array bounds for look-up table
    if(timeidx > tsize)
    {
        printf("timeidx=%d exceeds table size = %lu and range %d to %lu\n", timeidx, tsize, 0, tsize-1);
        exit(-1);
    }

    return DefaultProfile[timeidx];
}


// Simple linear interpolation example for table_accel(t) for any floating point t value
// for a table of accelerations that are 1 second apart in time, evenly spaced in time.
//
// accel[timeidx] <= accel[time] < accel[timeidx_next]
//
//
double faccel(double time)
{
    // The timeidx is an index into the known acceleration profile at a time <= time of interest passed in
    //
    // Note that conversion to integer truncates double to next lowest integer value or floor(time)
    //
    int timeidx = (int)time;

    // The timeidx_next is an index into the known acceleration profile at a time > time of interest passed in
    //
    // Note that the conversion to integer truncates double and the +1 is added for ceiling(time)
    //
    int timeidx_next = ((int)time)+1;

    // delta_t = time of interest - time at known value < time
    //
    // For more general case
    // double delta_t = (time - (double)((int)time)) / ((double)(timeidx_next - timeidx);
    //
    // If time in table is always 1 second apart, then we can simplify since (timeidx_next - timeidx) = 1.0 by definition here
    double delta_t = time - (double)((int)time);

    return ( 
               // The accel[time] is a linear value between accel[timeidx] and accel[timeidx_next]
               // 
               // The accel[time] is a value that can be determined by the slope of the interval and accel[timedix] 
               //
               // I.e. accel[time] = accel[timeidx] + ( (accel[timeidx_next] - accel[timeidx]) / ((double)(timeidx_next - timeidx)) ) * delta_t
               //
               //      ((double)(timeidx_next - timeidx)) = 1.0
               // 
               //      accel[time] = accel[timeidx] + (accel[timeidx_next] - accel[timeidx]) * delta_t
               //
               table_accel(timeidx) + ( (table_accel(timeidx_next) - table_accel(timeidx)) * delta_t)
           );
}
