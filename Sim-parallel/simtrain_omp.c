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


// Coefficient of Rolling Resistance from https://youtu.be/-KAVJH_Dl80
//
// Force_rolling_resist = m*accel
//
// Crr(m)(g) = m*accel
//
// accel = Crr(g)
//
// Assumes wheels don't lock up during braking or slip during acceleration
//
#define Crr_MIN (0.0003)
#define Crr_MAX (0.0004)
#define ACCEL_GRAVITY (9.81)

//double rolling_deceleration = Crr_MIN * ACCEL_GRAVITY;
double rolling_deceleration = 0.0;


// table look-up for acceleration profile given and velocity profile determined
double table_accel(int timeidx);
double table_vel(int timeidx);

// indirect generation of acceleration or velocity at any time with table interpolation
double faccel(double time);
double fvel(double time);

// Create velocity and position profiles (tables) the same size as acceleration profile
double VelProfile[sizeof(DefaultProfile) / sizeof(double)];
double PosProfile[sizeof(DefaultProfile) / sizeof(double)];

// Implement methods of integration for OpenMP
double Local_Riemann(double a, double b, int n, double func(double));
double Local_Trap(double a, double b, int n, double func(double));
double Local_Simpson(double a, double b, int n, double func(double));
double Local_RK4(double a, double b, int n, double func(double));

char *integrator_names[]={"Riemann", "Trapezoidal", "Simpson", "Runge-Kutta-4"};
#define RIEMANN 0
#define TRAPEZOIDAL 1
#define SIMPSON 2
#define RK4 3


void main(int argc, char *argv[])
{
    int idx;
    double time, dt=0.1; // dt=0.1 takes 10 steps per step in spreadsheet
    int tsize = (int)(sizeof(DefaultProfile) / sizeof(double));
    unsigned long integration_steps=tsize;
    int steps_per_idx=integration_steps/tsize;
    int thread_count=1, integrator_selected=0;
    double AccelStep, VelStep, PosStep;
    struct timespec start, end;
    double fstart, fend;


    printf("\nUse: simtrain [threads] [dt] [integrator is 0=Riemann, 1=Trap, 2=Simpsons, 3=RK4]\n");

    if(argc == 2)
    {
        sscanf(argv[1], "%d", &thread_count);
    }
    else if(argc == 3) 
    {
        sscanf(argv[1], "%d", &thread_count);
        sscanf(argv[2], "%lf", &dt);
    }
    else if(argc == 4) 
    {
        sscanf(argv[1], "%d", &thread_count);
        sscanf(argv[2], "%lf", &dt);
        sscanf(argv[3], "%d", &integrator_selected);
    }

    printf("\n***** Will simulate with %d threads, using dt=%lf, integrator=%s\n", thread_count, dt, integrator_names[integrator_selected]);

    integration_steps = (unsigned long) ((double)(tsize-1) / dt);
    steps_per_idx=integration_steps/(tsize-1);

    printf("\n***** Will use default time profile with thread_count=%d, with dt=%lf for %lu steps and %d steps per table entry\n",
           thread_count, dt, integration_steps, steps_per_idx);

    // Zero out VelProfile and PosProfile for next test
    for(idx=0; idx < tsize; idx++)
    {
        VelProfile[idx]=0.0;
        PosProfile[idx]=0.0;
    }

    // Integration to match spreadsheet with Look-up & interpolate integration function
    //
    // Potential to speed up with OpenMP or Pthreads
    //
    printf("\nTHREADED INTEGRATOR: integration with table with %d elements\n", tsize);
    clock_gettime(CLOCK_MONOTONIC, &start);
    VelStep=0.0; VelProfile[0]=VelStep;
    PosStep=0.0; PosProfile[0]=PosStep;
    double time_a, time_b;

    // Overall simulation table loop for time=0, to last time in model
    for(idx=0; idx < tsize-1; idx++)
    {
        time_a = (double)idx;
        time_b = (double)idx+1;

        switch(integrator_selected)
        {
            case RIEMANN:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_Riemann(time_a, time_b, steps_per_idx, faccel);
                VelProfile[idx+1]=VelStep;
                
                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_Riemann(time_a, time_b, steps_per_idx, fvel);
                PosProfile[idx+1]=PosStep;

                break;

            case TRAPEZOIDAL:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_Trap(time_a, time_b, steps_per_idx, faccel);
                VelProfile[idx+1]=VelStep;
                
                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_Trap(time_a, time_b, steps_per_idx, fvel);
                PosProfile[idx+1]=PosStep;

                break;


            case SIMPSON:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_Simpson(time_a, time_b, steps_per_idx, faccel);
                VelProfile[idx+1]=VelStep;
                
                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_Simpson(time_a, time_b, steps_per_idx, fvel);
                PosProfile[idx+1]=PosStep;

                break;

            case RK4:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_RK4(time_a, time_b, steps_per_idx, faccel);
                VelProfile[idx+1]=VelStep;
                
                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_RK4(time_a, time_b, steps_per_idx, fvel);
                PosProfile[idx+1]=PosStep;

                break;

            default:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_Riemann(time_a, time_b, steps_per_idx, faccel);
                VelProfile[idx+1]=VelStep;

                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_Riemann(time_a, time_b, steps_per_idx, fvel);
                PosProfile[idx+1]=PosStep;

                break;
        }
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
    double accel_input = table_accel(timeidx) + ( (table_accel(timeidx_next) - table_accel(timeidx)) * delta_t);

    // if train is speeding up, assume motor adds acceleration to overcome rolling deceleration
    if(accel_input > 0.0)
        return(accel_input + rolling_deceleration);

    // if train is braking, assume brakes are applied as needed over and above rolling deceleration
    else if(accel_input < 0.0)
        return(accel_input - rolling_deceleration);

    // if the train is coasting, then just return what should be no acceleration
    else
        return(accel_input);
}


double fvel(double time)
{
    int timeidx = (int)time;
    int timeidx_next = ((int)time)+1;
    double delta_t = time - (double)((int)time);

    return (table_vel(timeidx) + ( (table_vel(timeidx_next) - table_vel(timeidx)) * delta_t) );
}
