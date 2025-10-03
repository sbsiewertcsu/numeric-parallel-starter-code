#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include <mpi.h>

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

// 0.002943 to 0.003924 m/s²
double rolling_deceleration = Crr_MIN * ACCEL_GRAVITY;

double duration=1800.0;
double tscale, ascale, vscale; //computed in main based on actual duration

// direct generation of acceleration at any time with math library and arithmetic
double ex3_accel(double time);
double ex3_vel(double time);

// Implement methods of integration for OpenMP
double Local_Riemann(double a, double b, unsigned long n, double func(double));
double Local_Trap(double a, double b, unsigned long n, double func(double));
double Local_Simpson(double a, double b, unsigned long n, double func(double));
double Local_RK4(double a, double b, unsigned long n, double func(double));

char *integrator_names[]={"Riemann", "Trapezoidal", "Simpson", "Runge-Kutta-4"};
#define RIEMANN 0
#define TRAPEZOIDAL 1
#define SIMPSON 2
#define RK4 3

// mpiexec -n 4 ./simtrainideal 4 0.001 1800 0

void main(int argc, char *argv[])
{
    int idx;
    double time, dt=1.0; // dt=1.0 is the default to match spreadsheet
    unsigned long integration_steps;
    int integrator_selected=0;
    int thread_count=4;
    double AccelStep, VelStep, PosStep;
    struct timespec start, end;
    double fstart, fend;
    double TargetPos=122000.0;
    double targetErr=0.0;
    double leastErr=0.0;

    struct {
        double posErr;
        int rank;
        } global_err;

    int comm_sz;
    int my_rank, best_rank;
    double aveVel=0.0;
    double distLeft=0.0;
    double estTime = 0.0;
    double time_a, time_b;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0) printf("\nUse: simtrain [threads] [dt] [duration] [integrator is 0=Riemann, 1=Trap, 2=Simpsons, 3=RK4]\n");

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
        sscanf(argv[3], "%lf", &duration);
    }
    else if(argc == 5) 
    {
        sscanf(argv[1], "%d", &thread_count);
        sscanf(argv[2], "%lf", &dt);
        sscanf(argv[3], "%lf", &duration);
        sscanf(argv[4], "%d", &integrator_selected);
    }

    integration_steps = duration / dt;

    // determined such that the sine curve is stretched over duration
    //tscale=1.0;
    tscale=duration/(2.0*M_PI);

    //ascale=1.0;
    ascale=0.2365893166123-rolling_deceleration;

    //vscale=1.0;
    vscale=ascale*duration/(2.0*M_PI);


    // Rank 0, runs a full simulation which wil come up short of the target distance
    //
    // This is used to estimate time to complete distance based on average velocity for the trial run
    //
    // All ranks will evenly divide the time between the original schedule and the estimated addtional time
    //
    // Reduce to find minimum error between the target distance and each rank's simulation distance will be used
    // to determine the closest time to the actual time required given the rolling resistance.
    //
    if(my_rank == 0)
    {
        printf("Will simulate with thread_count=%d, with dt=%lf for %lu steps for %lf seconds with integrator %s\n",
               thread_count, dt, integration_steps, duration, integrator_names[integrator_selected]);

        printf("\n\nTHREADED INTEGRATOR %s: test for duration %lf seconds\n", integrator_names[integrator_selected], duration);
        clock_gettime(CLOCK_MONOTONIC, &start);
        VelStep=0.0; PosStep=0.0;

        // Integrate the whole simulation in parallel based upon Oracle antiderivative

        time_a = 0.0;
        time_b = duration;

        switch(integrator_selected)
        {
            case RIEMANN:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_Riemann(time_a, time_b, integration_steps, ex3_accel);

                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_Riemann(time_a, time_b, integration_steps, ex3_vel);

                break;

            case TRAPEZOIDAL:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_Trap(time_a, time_b, integration_steps, ex3_accel);

                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_Trap(time_a, time_b, integration_steps, ex3_vel);

                break;


            case SIMPSON:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_Simpson(time_a, time_b, integration_steps, ex3_accel);

                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_Simpson(time_a, time_b, integration_steps, ex3_vel);

                break;

            case RK4:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_RK4(time_a, time_b, integration_steps, ex3_accel);

                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_RK4(time_a, time_b, integration_steps, ex3_vel);

                break;

            default:
                #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
                VelStep += Local_Riemann(time_a, time_b, integration_steps, ex3_accel);

                #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
                PosStep += Local_Riemann(time_a, time_b, integration_steps, ex3_vel);
        }


        clock_gettime(CLOCK_MONOTONIC, &end);
        fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
        fend=end.tv_sec + (end.tv_nsec / 1000000000.0);

        aveVel=PosStep/duration;
        distLeft=TargetPos-PosStep;
        estTime = distLeft / aveVel;

        printf("Train from function in %lf seconds: final velocity = %lf, final position = %lf, ave velocity=%lf, remaining dist=%lf, added time=%lf\n",
               (fend-fstart), VelStep, PosStep, aveVel, distLeft, estTime);

    } // end my_rank==0

    // Wait for rank 0 to finish, then go on to simulate multiple cases in parallel
    //
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&estTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Divide up duration search space between oringal duration and duration+estTime
    duration=duration + (estTime*(double)((double)(my_rank+1)/(double)comm_sz));
    printf("rank %d of %d, will run simulation for %lf time\n", my_rank, comm_sz, duration);

    // Start parallel simulation here
    //
    integration_steps = duration / dt;
    time_a = 0.0;
    time_b = duration;

    tscale=duration/(2.0*M_PI);
    vscale=ascale*duration/(2.0*M_PI);


    printf("rank %d will simulate with thread_count=%d, with dt=%lf for %lu steps from a=%lf to b=%lf, for %lf seconds with integrator %s\n",
           my_rank, thread_count, dt, integration_steps, time_a, time_b, duration, integrator_names[integrator_selected]);

    printf("\n\nTHREADED INTEGRATOR %s: test for duration %lf seconds\n", integrator_names[integrator_selected], duration);
    clock_gettime(CLOCK_MONOTONIC, &start);

    VelStep=0.0; PosStep=0.0;

    // Integrate the whole simulation in parallel based upon Oracle antiderivative

    switch(integrator_selected)
    {
        case RIEMANN:
            #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
            VelStep += Local_Riemann(time_a, time_b, integration_steps, ex3_accel);

            #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
            PosStep += Local_Riemann(time_a, time_b, integration_steps, ex3_vel);

            break;

        case TRAPEZOIDAL:
            #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
            VelStep += Local_Trap(time_a, time_b, integration_steps, ex3_accel);

            #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
            PosStep += Local_Trap(time_a, time_b, integration_steps, ex3_vel);

            break;


        case SIMPSON:
            #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
            VelStep += Local_Simpson(time_a, time_b, integration_steps, ex3_accel);

            #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
            PosStep += Local_Simpson(time_a, time_b, integration_steps, ex3_vel);

            break;

        case RK4:
            #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
            VelStep += Local_RK4(time_a, time_b, integration_steps, ex3_accel);

            #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
            PosStep += Local_RK4(time_a, time_b, integration_steps, ex3_vel);

            break;

        default:
            #pragma omp parallel num_threads(thread_count) reduction(+:VelStep)
            VelStep += Local_Riemann(time_a, time_b, integration_steps, ex3_accel);

            #pragma omp parallel num_threads(thread_count) reduction(+:PosStep)
            PosStep += Local_Riemann(time_a, time_b, integration_steps, ex3_vel);
    }


    clock_gettime(CLOCK_MONOTONIC, &end);
    fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
    fend=end.tv_sec + (end.tv_nsec / 1000000000.0);

    aveVel=PosStep/duration;
    distLeft=TargetPos-PosStep;
    estTime = distLeft / aveVel;
    targetErr = fabs(TargetPos - PosStep);

    global_err.posErr=targetErr;
    global_err.rank=my_rank;

    printf("Rank %d, simulated train from function in %lf seconds: final velocity = %lf, final position = %lf, ave velocity=%lf, remaining dist=%lf, added time=%lf\n",
           my_rank, (fend-fstart), VelStep, PosStep, aveVel, distLeft, estTime);

    MPI_Allreduce(&targetErr, &leastErr, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_err, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    if(my_rank == 0) 
    {
        printf("rank = %d has leastErr=%lf, leastErr=%lf\n", global_err.rank, global_err.posErr, leastErr);
    }

    MPI_Finalize();

}


double Local_Riemann(double a, double b, unsigned long n, double funct(double))
{
    double dt, interval_sum=0.0, local_a, local_b, time;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    dt = (b-a)/((double)n);

    unsigned long idx, local_n;

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


double Local_Trap(double a, double b, unsigned long n, double funct(double))
{
    double dt = (b - a) / n;
    double local_a, local_b, time, interval_sum;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    unsigned long idx, local_n;

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


double Local_Simpson(double a, double b, unsigned long n, double funct(double))
{
    double dt = (b - a) / n;
    double interval_sum = 0.0;
    double local_a, local_b, time, fx;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    unsigned long idx, local_n;

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


double Local_RK4(double a, double b, unsigned long n, double funct(double))
{
    double dt = (b - a) / n;
    double interval_sum = 0.0;
    double local_a, local_b, time, k1, k2, k3, k4;
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    unsigned long idx, local_n;

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


double ex3_accel(double time)
{
    return (sin(time/tscale)*ascale);
}


// determined based on known anti-derivative of ex4_accel function
double ex3_vel(double time)
{
    return ((-cos(time/tscale)+1)*vscale);
}
