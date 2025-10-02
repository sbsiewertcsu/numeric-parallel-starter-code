#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// For values between 1 second indexed data, use linear interpolation to determine profile value at any "t".
//
// Wikipedia - https://en.wikipedia.org/wiki/Linear_interpolation
//
// This assumes that the profile is "piecewise linear", which is accurate for linear or constant functions, and 
// reasonably accurate for most non-linear functions over small intervals of 1 second.
//
// For high frequency non-linear functions spline and other more advanced curve fitting methods could be used, but
// the piecewise linear assumption is sufficient for CSCI 551 for acceleration profiles given.
//
// The alternative to a look-up table with linear interpolation is direct modeling of a function, but this requires 
// knowledge of the profile function as a linear, polynomial, or transcendental function or combination there-of, and
// this may not be known.

#include "ex3.h"
//#include "ex4.h"
//#include "const.h"
//#include "sine.h"


// table look-up for acceleration profile given and velocity profile determined
double table_accel(int timeidx);
double table_vel(int timeidx);

// indirect generation of acceleration or velocity at any time with table interpolation
double faccel(double time);
double fvel(double time);

// direct generation of acceleration at any time with math library and arithmetic
double ex3_accel(double time);
double ex3_vel(double time);
double ex4_accel(double time);

// Create velocity and position profiles (tables) the same size as acceleration profile
double VelProfile[sizeof(DefaultProfile) / sizeof(double)];
double PosProfile[sizeof(DefaultProfile) / sizeof(double)];

// Implement methods of integration for OpenMP
double Local_Riemann(double a, double b, int n, double func(double));
double Local_Trap(double a, double b, int n, double func(double));
double Local_Simpson(double a, double b, int n, double func(double));
double Local_RK4(double a, double b, int n, double func(double));


void main(int argc, char *argv[])
{
    int idx;
    double time, dt=1.0; // dt=1.0 is the default to match spreadsheet
    int tsize = (int)(sizeof(DefaultProfile) / sizeof(double));
    unsigned long integration_steps=tsize;
    int steps_per_idx=integration_steps/tsize;
    int thread_count=1;
    double AccelStep, VelStep, PosStep;
    struct timespec start, end;
    double fstart, fend;


    printf("argc=%d, argv[0]=%s\n", argc, argv[0]);

    if(argc == 2)
    {
        sscanf(argv[1], "%d", &thread_count);
    }
    else if(argc == 3) 
    {
        sscanf(argv[1], "%d", &thread_count);
        sscanf(argv[2], "%lf", &dt);
    }
    else
    {
        printf("Will match the spreadsheet design\n");
    }

    integration_steps = (unsigned long) ((double)(tsize-1) / dt);
    steps_per_idx=integration_steps/(tsize-1);

    printf("Will use default time profile with thread_count=%d, with dt=%lf for %lu steps and %d steps per table entry\n",
           thread_count, dt, integration_steps, steps_per_idx);

    // Zero out VelProfile and PosProfile for next test
    for(idx=0; idx < tsize; idx++)
    {
        VelProfile[idx]=0.0;
        PosProfile[idx]=0.0;
    }

    // Left Riemann sum test to match spreadsheet with abstracted integration function
    //
    // Potential to speed up with OpenMP or Pthreads
    //
    printf("\n\nTHREADED INTEGRATOR FUNCTION: Left Riemann sum test for table with %d elements\n", tsize);
    clock_gettime(CLOCK_MONOTONIC, &start);
    VelStep=0.0; VelProfile[0]=VelStep;
    PosStep=0.0; PosProfile[0]=PosStep;
    double time_a, time_b;

    // Overall simulation table loop
    for(idx=1; idx < tsize; idx++)
    {
        time_a = (double)idx-1;
        time_b = (double)idx;

        #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
        //VelStep += Local_Riemann(time_a, time_b, steps_per_idx, faccel);
        //VelStep += Local_Trap(time_a, time_b, steps_per_idx, faccel);
        //VelStep += Local_Simpson(time_a, time_b, steps_per_idx, faccel);
        //VelStep += Local_RK4(time_a, time_b, steps_per_idx, faccel);
        VelStep += Local_RK4(time_a, time_b, steps_per_idx, ex3_accel);

        //printf("VelStep=%lf at time=%lf\n", VelStep, time_b);
        VelProfile[idx]=VelStep;

        #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
        //PosStep += Local_Riemann(time_a, time_b, steps_per_idx, fvel);
        //PosStep += Local_Trap(time_a, time_b, steps_per_idx, fvel);
        //PosStep += Local_Simpson(time_a, time_b, steps_per_idx, fvel);
        //PosStep += Local_RK4(time_a, time_b, steps_per_idx, fvel);
        PosStep += Local_RK4(time_a, time_b, steps_per_idx, ex3_vel);

        //printf("PosStep=%lf at time=%lf\n", PosStep, time_b);
        PosProfile[idx]=PosStep;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
    fend=end.tv_sec + (end.tv_nsec / 1000000000.0);

    printf("final table index = %d for table of size %d\n", idx, tsize);
    printf("Train from table in %lf seconds with %d samples: final velocity = %lf, final position = %lf\n", 
	       (fend-fstart), tsize, VelProfile[tsize-1], PosProfile[tsize-1]);

}


double Local_Riemann(double a, double b, int n, double funct(double))
{
    double dt, interval_sum=0.0, local_a, local_b, time;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    dt = (b-a)/((double)n);

    int idx, local_n;

    local_n = n / thread_count;

    local_a = a + my_rank*local_n*dt;
    local_b = local_a + local_n*dt;


    for(idx=1; idx <= local_n; idx++)
    {
        time = local_a + idx*dt;
        interval_sum += (funct(time) * dt);
        //printf("Step for my_rank=%d at time=%lf, f(t)=%lf, sum=%lf\n", my_rank, time, funct(time), interval_sum);
    }

    //printf("Local Riemann = %lf for my_rank=%d of threads %d with dt=%lf, on a=%lf to b=%lf for %d steps\n",
    //        interval_sum, my_rank, thread_count, dt, local_a, local_b, local_n);

    return interval_sum;
}


double Local_Trap(double a, double b, int n, double funct(double))
{
    double dt = (b - a) / n;
    double local_a, local_b, time, interval_sum;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    int idx, local_n;

    local_n = n / thread_count;

    local_a = a + my_rank*local_n*dt;
    local_b = local_a + local_n*dt;

    interval_sum = (funct(local_a) + funct(local_b)) / 2.0;

    for (idx = 1; idx < local_n; idx++)
    {
        time = local_a + idx * dt;
        interval_sum += funct(time);
    }

    return dt * interval_sum;
}


double Local_Simpson(double a, double b, int n, double funct(double))
{
    double dt = (b - a) / n;
    double interval_sum = 0.0;
    double local_a, local_b, time, fx;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    int idx, local_n;

    local_n = n / thread_count;

    local_a = a + my_rank*local_n*dt;
    local_b = local_a + local_n*dt;

    for (idx = 1; idx <= local_n; idx++)
    {
        time = local_a + idx * dt;
        fx = funct(time);

        // See https://en.wikipedia.org/wiki/Simpson's_rule for more information
        //
        // 1) on the first we evaluate f(a) and on the last we evaluate f(b)
        // 2) for steps between we alternate between (4/3)f(a+b) and (2/3)f(a+b)
        // 3) at the end we return h times the weighted sum of all the 1/3, 4/3, 2/3 summation
        //    terms.
        //
        if (idx == 0 || idx == local_n)
        {
            interval_sum += fx;
        }

        // Alternating 4/3 and 2/3 weighting for points between f(a) and f(b)
        else if (idx % 2 == 1)
        {
            interval_sum += 4.0 * fx;
        }
        else
        {
            interval_sum += 2.0 * fx;
        }
    }

    // h=(b-a)/n, sum= [f(a) + f(b)] + (4 or 2)*f(a+b), all multipied by 1/3
    return dt * interval_sum / 3.0;
}


double Local_RK4(double a, double b, int n, double funct(double))
{
    double dt = (b - a) / n;
    double interval_sum = 0.0;
    double local_a, local_b, time, k1, k2, k3, k4;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    int idx, local_n;

    local_n = n / thread_count;

    local_a = a + my_rank*local_n*dt;
    local_b = local_a + local_n*dt;


    // March from a to b in n uniform steps
    for (idx = 1; idx <= local_n; idx++)
    {
        time = local_a + idx * dt;

        // RK4 stages: k1..k4 evaluate f at time, time+dt/2, time+dt
        k1 = funct(time);
        k2 = funct(time + 0.5 * dt);
        k3 = funct(time + 0.5 * dt);
        k4 = funct(time + dt);

        // Update the integral using RK4 combination
        interval_sum += k1 + 2.0*k2 + 2.0*k3 + k4;
    }

    return dt * interval_sum / 6.0;
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


double table_vel(int timeidx)
{
    long unsigned int tsize = sizeof(VelProfile) / sizeof(double);

    if(timeidx > tsize)
    {
        printf("timeidx=%d exceeds table size = %lu and range %d to %lu\n", timeidx, tsize, 0, tsize-1);
        exit(-1);
    }

    return VelProfile[timeidx];
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


double fvel(double time)
{
    int timeidx = (int)time;
    int timeidx_next = ((int)time)+1;
    double delta_t = time - (double)((int)time);

    return (table_vel(timeidx) + ( (table_vel(timeidx_next) - table_vel(timeidx)) * delta_t) );
}


double ex3_accel(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that acceleration will peak to result in translation of 122,000.0 meters
    static double ascale=0.2365893166123;

    return (sin(time/tscale)*ascale);
}


// determined based on known anti-derivative of ex4_accel function
double ex3_vel(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static double vscale=0.2365893166123*1800.0/(2.0*M_PI);

    return ((-cos(time/tscale)+1)*vscale);
}


double ex4_accel(double time)
{
    // maximum acceleration and deceleration in m/s/s
    static double peak_accel=0.2905;

    // model each phase of the acceleration profile directly
    
    // linearly increasing acceleration by 0.01 m/s/s up to peak_accel
    if(time >= 0.0 && time <= 100.0)
    {
        return ( (peak_accel/100.0)*time );
    }

    // constant acceleration at peak_accel
    if(time > 100.0 && time <= 300.0)
    {
	return peak_accel;
    }
    
    // linearly decreasing acceleration by -0.01 m/s/s to zero
    if(time > 300.0 && time <= 400.0)
    {
        return ( peak_accel - ((peak_accel/100.0)*(time-300.0)) );
    }
   
    // zero acceleration
    if(time > 400.0 && time <= 1400.0)
    {
	return 0.0;
    }

    // linearly decreasing acceleration by -0.01 m/s/s to -peak_accel
    if(time > 1400.0 && time <= 1500.0)
    {
        return ( -((peak_accel/100.0)*(time-1400.0)) );
    }
    
    // constant acceleration at -peak_accel
    if(time > 1500.0 && time <= 1700.0)
    {
        return -peak_accel;	
    }
    
    // linearly increasing acceleration by 0.01 up to zero
    if(time > 1700.0 && time <= 1800.0)
    {
        return ( (peak_accel/100.0)*(time-1700.0) );
    }
}
