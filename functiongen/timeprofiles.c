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
// For high frequency non-linear functions spline and other more advanced curve fitting methods could be used, but
// the piecewise linear assumption is sufficient for CSCI 551 for acceleration profiles given.
//
// The alternative to a look-up table with linear interpolation is direct modeling of a function, but this requires 
// knowledge of the profile function as a linear, polynomial, or transcendental function or combination there-of, and
// this may not be known.

//#include "ex3accel.h"
#include "ex3.h"
//#include "ex4.h"
//#include "ex6lin.h"
//#include "ex6nonlin.h"


// table look-up for acceleration profile given and velocity profile determined
double table_accel(int timeidx);
double table_vel(int timeidx);

// indirect generation of acceleration or velocity at any time with table interpolation
double faccel(double time);
double fvel(double time);

// direct generation of acceleration at any time with math library and arithmetic
double ex3_accel(double time);
double ex4_accel(double time);
double ex4_vel(double time);

// Create velocity and position profiles (tables) the same size as acceleration profile
double VelProfile[sizeof(DefaultProfile) / sizeof(double)];
double PosProfile[sizeof(DefaultProfile) / sizeof(double)];

// Implement methods of integration
double Local_Riemann(double a, double b, int n, double func(double));
double Local_Trap(double a, double b, int n, double func(double));


void main(int argc, char *argv[])
{
    int idx;
    double time, dt=1.0; // dt=1.0 is the default to match spreadsheet
    int tsize = (int)(sizeof(DefaultProfile) / sizeof(double));
    unsigned long integration_steps=tsize;
    int steps_per_idx=integration_steps/tsize;
    unsigned long istep=0;
    int thread_count=2;
    double AccelStep, VelStep, PosStep;
    struct timespec start, end;
    double fstart, fend;


    printf("argc=%d, argv[0]=%s\n", argc, argv[0]);

    if(argc == 2)
    {
        sscanf(argv[1], "%d", &thread_count);
        printf("Use: trainprofile [thread_count] [dt]\n");
        printf("Will use default time profile and use thread_count=%d\n", thread_count);
    }
    else if(argc == 3) 
    {
        sscanf(argv[1], "%d", &thread_count);
        sscanf(argv[2], "%lf", &dt);
        integration_steps = (unsigned long) ((double)tsize / dt);
        steps_per_idx=integration_steps/tsize;
        printf("Use: trainprofile [thread_count] [dt]\n");
        printf("Will use default time profile and use thread_count=%d, with dt=%lf for %lu steps and %d steps per table entry\n", thread_count, dt, integration_steps, steps_per_idx);
    }
    else
    {
        printf("Use: trainprofile [thread_count] [dt]\n");
        printf("Will use default time profile\n");
    }

    // Verify the static initializer or loaded CSV array for size and values
    //
    // Test outputs here should match the original spreadsheet specificaiton for the acceleration
    // profile.
    //
    // This is just a simple spot-checking test of expected values
    //
    printf("Number of values in profile = %lu for 1801 expected\n", sizeof(DefaultProfile)/sizeof(double));

    // Test interpolation between 0...1
    printf("A[%d]=%015.14lf\n", 0, table_accel(0));
    for(idx=0; idx <= 10; idx++)
    {
        time = 0.0 + (0.1*(double)idx);
        printf("\tA[%015.14lf]=%015.14lf\n", time, faccel(time));
    }
    printf("A[%d]=%015.14lf\n", 1, table_accel(1));


    // Test interpolation between 1400...1401
    printf("A[%d]=%015.14lf\n", 1400, table_accel(1400));
    for(idx=0; idx <= 10; idx++)
    {
        time = 1400.0 + (0.1*(double)idx);
        printf("\tA[%015.14lf]=%015.14lf\n", time, faccel(time));
    }
    printf("A[%d]=%015.14lf\n", 1401, table_accel(1401));


    // Test interpolation between 1799...1800
    printf("A[%d]=%015.14lf\n", 1799, table_accel(1799));
    printf("A[%lf]=%015.14lf\n", 1799.5, faccel(1799.5));
    printf("A[%d]=%015.14lf\n", 1800, table_accel(1800));

    // Test access outside of array bounds
    //printf("A[%d]=%015.14lf\n", 1801, table_accel(1801));
    //printf("A[%d]=%015.14lf\n", 1801, table_accel(1802));


    // Left Riemann sum test to match spreadsheet with shared data approach
    printf("\n\nLeft Riemann sum test for table with %d elements\n", tsize);
    clock_gettime(CLOCK_MONOTONIC, &start);
    VelProfile[0] = 0.0; VelStep=0;
    PosProfile[0] = 0.0; PosStep=0;
    idx=0;


    // **** SIMPLE SEQUENTIAL SHARED DATA INTEGRATION
    //
    // Hard to speed up due to data dependencies.
    //
    for(istep=1; istep < integration_steps; istep++)
    {
	    // for Left Riemann take acceleration at current time and add to previous value
	    time = (double)istep * dt;

	    // Note that velocity can be determined from profile or from math model for acceleration
        AccelStep = faccel(time);

        VelStep = VelStep + (AccelStep * dt);
       
        if((istep % steps_per_idx) == 0) 
        {
            idx++;
            VelProfile[idx] = VelStep;
        }

	    // Note that position must be determined from velocity determined above
        PosStep = PosStep + (VelStep * dt);

        if((istep % steps_per_idx) == 0) 
        {
            PosProfile[idx] = PosStep;
	        //printf("cond=%d, tsize=%d, idx=%d, istep=%lu, time=%lf, accel=%lf, vel=%lf, pos=%lf\n", 
		    //       (istep<integration_steps), tsize, idx, istep, time, faccel(time), VelProfile[idx], PosProfile[idx]);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
    fend=end.tv_sec + (end.tv_nsec / 1000000000.0);

    printf("Train from table in %lf seconds with %d samples: final velocity = %lf, final position = %lf\n", 
	       (fend-fstart), tsize, VelProfile[tsize-1], PosProfile[tsize-1]);

    //exit(0);


    // Zero out VelProfile and PosProfile for next test
    for(idx=0; idx < tsize; idx++)
    {
        VelProfile[idx]=0.0;
        PosProfile[idx]=0.0;
    }



    // **** 2 PART SEQUENTIAL SHARED DATA INTEGRATION
    //
    // Left Riemann sum test to match spreadsheet with integration in 2 parts
    // to suggest how one might use MPI with ranks to divide up work.
    //

    // Rank 0 would do the left half of the integration 
    printf("\n\nLeft Riemann sum test for half-table with %d elements and %lu steps\n", tsize/2, integration_steps/2);
    VelProfile[0] = 0.0; VelStep=0;
    PosProfile[0] = 0.0; PosStep=0;
    idx=0;

    for(istep=1; istep <= integration_steps/2; istep++)
    {
	    // for Left Riemann take acceleration at current time and add to previous value
	    time = (double)istep * dt;

	    // Note that velocity can be determined from profile or from math model for acceleration
        AccelStep = faccel(time);
        VelStep += (AccelStep * dt);
       
        if((istep % steps_per_idx) == 0) 
        {
            idx++;
            VelProfile[idx] = VelStep;
        }

	    // Note that position must be determined from velocity determined above
        PosStep += (VelStep * dt);

        if((istep % steps_per_idx) == 0) 
        {
            PosProfile[idx] = PosStep;
	        //printf("cond=%d, tsize=%d, idx=%d, istep=%lu, time=%lf, accel=%lf, vel=%lf, pos=%lf\n", 
		    //       (istep<integration_steps), tsize, idx, istep, time, faccel(time), VelProfile[idx], PosProfile[idx]);
        }

    }


    // Rank 1 would do the right half of the integration but we must adjust for starting conditions
    //
    // The problem is that we will not know the starting conditions until Rank 0 is done, which can be done
    // with MPI, but would serialize the ranks and provide no speed-up.
    //
    time = (double)(tsize/2);
    VelProfile[tsize/2] = VelStep;        // Vel[start]
    PosProfile[tsize/2] = PosStep;        // Pos[start]
    idx=tsize/2;

    printf("Left Riemann sum test for half-table with %d elements and %lu steps starting at time=%lf\n", tsize/2, integration_steps/2, time);

    for(istep=(integration_steps/2)+1; istep < integration_steps; istep++)
    {
	    // for Left Riemann take acceleration at current time and add to previous value
	    time = (double)istep * dt;

	    // Note that velocity can be determined from profile or from math model for acceleration
        AccelStep = faccel(time);
        VelStep += (AccelStep * dt);
       
        if((istep % steps_per_idx) == 0) 
        {
            idx++;
            VelProfile[idx] = VelStep;
        }

	    // Note that position must be determined from velocity determined above
        PosStep += (VelStep * dt);

        if((istep % steps_per_idx) == 0) 
        {
            PosProfile[idx] = PosStep;
	        //printf("cond=%d, tsize=%d, idx=%d, istep=%lu, time=%lf, accel=%lf, vel=%lf, pos=%lf\n", 
		    //       (istep<integration_steps), tsize, idx, istep, time, faccel(time), VelProfile[idx], PosProfile[idx]);
        }

    }

    printf("Train from table with %d samples: final velocity = %lf, final position = %lf\n", 
	    tsize, VelProfile[tsize-1], PosProfile[tsize-1]);


    exit(0);

    // Zero out VelProfile and PosProfile for next test
    for(idx=0; idx < tsize; idx++)
    {
        VelProfile[idx]=0.0;
        PosProfile[idx]=0.0;
    }


    // **** 2 PART, 2 PHASE SEQUENTIAL SHARED DATA INTEGRATION
    //
    // Left Riemann sum test to match spreadsheet with integration PHASES in 2 parts
    // to suggest how one might use MPI with ranks to divide up work.
    //
    // This provides speed-up because I can do all velocity integration in parallel
    //
    // SYNC UP and adjust velocity V[0]
    //
    // Then do all position integration in parallel
    //
    // SYNC UP and adjust postion P[0]
    //
    // Provide final output and/or combined profiles collected by each rank.
    //


    // PARALLEL BEGIN
    //
    // Rank 0 would do the left half of the integration as before but just acceleration first
    printf("\n\nLeft Riemann VELOCITY sum test for half-table with %d elements and %lu steps\n", tsize/2, integration_steps/2);
    VelProfile[0] = 0.0; VelStep=0.0;
    idx=0;

    for(istep=1; istep <= integration_steps/2; istep++)
    {
	    time = (double)istep * dt;
        AccelStep = faccel(time);
        VelStep += (AccelStep * dt);
       
        if((istep % steps_per_idx) == 0) 
        {
            idx++;
            VelProfile[idx] = VelStep;
	        //printf("cond=%d, tsize=%d, idx=%d, time=%lf, accel=%lf, vel=%lf\n", 
            //		(idx<tsize), tsize, idx, time, faccel(time), VelProfile[idx]);
        }
    }


    // Rank 1 would do the right half of the integration but we must adjust for starting conditions
    printf("\nLeft Riemann VELOCITY sum test for half-table with %d elements and %lu steps\n", tsize/2, integration_steps/2);
    VelProfile[tsize/2] = 0 + (faccel(time) * dt); VelStep=0.0;
    idx=tsize/2;


    for(istep=(integration_steps/2)+1; istep < integration_steps; istep++)
    {
	    time = (double)istep * dt;
        AccelStep = faccel(time);
        VelStep += (AccelStep * dt);

        if((istep % steps_per_idx) == 0) 
        {
            idx++;
            VelProfile[idx] = VelStep;
	        //printf("cond=%d, tsize=%d, idx=%d, time=%lf, accel=%lf, vel=%lf\n", 
            //		(idx<tsize), tsize, idx, time, faccel(time), VelProfile[idx]);
        }
    }
    //
    // PARALLEL END



    // Make adjustment to final velocity at end of each rank integration completion
    // SERIAL or use OpenMP to speed-up perhaps, but may not be worth it, depends on the duration of the simulation
    //
    for(idx=(tsize/2); idx < tsize; idx++)
        VelProfile[idx] +=  VelProfile[(tsize/2)-1];

    printf("Train from table with %d samples: final velocity = %lf\n", tsize, VelProfile[tsize-1]);



    // PARALLEL BEGIN
    //
    // Rank 0 would do the left half of the integration as before but just velocity now
    printf("\n\nLeft Riemann POSITION sum test for half-table with %d elements and %lu steps\n", tsize/2, integration_steps/2);
    PosProfile[0] = 0.0; PosStep=0;
    idx=0;

    for(istep=1; istep <= integration_steps/2; istep++)
    {
	    time = (double)istep * dt;
        VelStep = fvel(time);
        PosStep += (VelStep * dt);

        if((istep % steps_per_idx) == 0) 
        {
            idx++;
            PosProfile[idx] = PosStep;
	        // printf("cond=%d, tsize=%d, idx=%d, time=%lf, accel=%lf, pos=%lf\n", 
	        // 	(idx<tsize), tsize, idx, time, faccel(time), PosProfile[idx]);
        }
    }


    // Rank 1 would do the right half of the integration but we must adjust for starting conditions
    printf("Left Riemann VELOCITY sum test for half-table with %d elements\n", tsize/2);
    PosProfile[tsize/2] = 0 + (fvel(time) * dt); PosStep=0.0;
    idx=tsize/2;

    for(istep=(integration_steps/2)+1; istep < integration_steps; istep++)
    {
	    time = (double)istep * dt;
        VelStep = fvel(time);
        PosStep += PosStep + (VelStep * dt);

        if((istep % steps_per_idx) == 0) 
        {
            idx++;
            PosProfile[idx] = PosStep;
	        // printf("cond=%d, tsize=%d, idx=%d, time=%lf, accel=%lf, pos=%lf\n", 
		    //       (idx<tsize), tsize, idx, time, faccel(time), PosProfile[idx]);
        }
    }
    //
    // PARALLEL END


    // Make adjustment to final position at end of each rank integration completion
    // SERIAL or use OpenMP to speed-up perhaps, but may not be worth it, depends on the duration of the simulation
    for(idx=(tsize/2); idx < tsize; idx++)
        PosProfile[idx] +=  PosProfile[(tsize/2)-1];


    printf("Train from table with %d samples: final velocity = %lf, final position = %lf\n", 
	    tsize, VelProfile[tsize-1], PosProfile[tsize-1]);
}


double Local_Riemann(double a, double b, int n, double funct(double))
{
    double dt, interval_sum=0.0, local_a, local_b, time;
    int my_rank;
    int rank_count;

    dt = (b-a)/((double)n);

    int idx, local_n;

    local_n = n / rank_count;

    local_a = a + my_rank*local_n*dt;
    local_b = local_a + local_n*dt;


    for(idx=1; idx <= local_n; idx++)
    {
        time = local_a + idx*dt;
        interval_sum += (funct(time) * dt);
        //printf("Step for my_rank=%d at time=%lf, f(t)=%lf, sum=%lf\n", my_rank, time, funct(time), interval_sum);
    }

    //printf("Local Riemann = %lf for my_rank=%d of rank_count %d with dt=%lf, on a=%lf to b=%lf for %d steps\n",
    //        interval_sum, my_rank, rank_count, dt, local_a, local_b, local_n);

    return interval_sum;
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


double ex4_accel(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that acceleration will peak to result in translation of 122,000.0 meters
    static double ascale=0.2365893166123;

    return (sin(time/tscale)*ascale);
}


// determined based on known anti-derivative of ex4_accel function
double ex4_vel(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static double vscale=0.2365893166123*1800.0/(2.0*M_PI);

    return ((-cos(time/tscale)+1)*vscale);
}

