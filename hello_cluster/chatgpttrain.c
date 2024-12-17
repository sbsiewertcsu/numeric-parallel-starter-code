// Request issued to https://chat.openai.com/chat
//
// Write an MPI program that integrates the acceleration data in this file to generate
// final velocity and position - https://www.ecst.csuchico.edu/~sbsiewert/csci551/code/functiongen/ex4.h
//
// To integrate the acceleration data in the provided file and generate final velocity and position using MPI, we
// can follow these steps:
//
//    1) Read the acceleration data from the file and distribute it evenly among the processes.
//
//    2) Each process calculates the velocity and position for its own data segment using numerical integration.
//       We can use the trapezoidal rule for numerical integration. The trapezoidal rule approximates the area under
//       the curve of acceleration as a trapezoid with the top parallel to the x-axis and the sides as the acceleration
//       at the start and end of the interval.
//
//    3) Each process sends its final velocity and position to the master process.
//
//    4) The master process calculates the final velocity and position by adding the contributions from all the processes.
//
// Here's the MPI program in C to accomplish the above steps, but it does not scale:
// 
// ChatGPT failed at producing code that would compile (diappointing), but some of the fixes are fairly obvious:
//
//    1) read_data is not necssary - we have a static initializer that already fills out the table, so set 
//       acceleration pointer to DefaultProfile array (we asked ChatGPT to use this?)
//
//    2) SAMPLE_RATE is not defined - it should be 1.0 here since the data is for a dt=1.0 (with no interpolation)
// 
//    3) Forgot to include stdlid.h needed for "free"
//
//    4) Set num_samples to the size of the static initializer DefaultProfile - read_data is a figment of ChatGPT's
//       imagination.
//
//    5) After the above fixes, the code built and ran and in fact produced the expected velocity (0.0) and position (122Km),
//       but it dumped core when it called free:
//
//       sbsiewert@o251-14:~/public_html/csci551/code/hello_cluster$ ./chatgpttrain
//       Final velocity: -0.000000
//       Final position: 122000.004000
//       free(): invalid pointer
//       Aborted (core dumped)
//
//       Fix was to comment out the "free" - it was never allocated (it assumed that read_data did this?).
//
//     6) Like compare.c and the Excel spreadsheet model, this code takes steps that are equal to the table step size
//        only.  It does not interpolate the look-up table for integration with a smaller step size and it does not save the
//        velocity profile (a new table with computed velocity as a function of time).
//
//     7) However, unlike compare.c, it does not actually at all for more than 1 worker!  Sorry ChatGPT, you failed!
//        Here's what we see if we scale it up:
//
//        sbsiewert@o251-01:~/public_html/csci551/code/hello_cluster$ mpiexec -n 1 ./chatgpttrain
//        Final velocity: -0.000000
//        Final position: 122000.004000
//        sbsiewert@o251-01:~/public_html/csci551/code/hello_cluster$ mpiexec -n 2 ./chatgpttrain
//        Final velocity: 0.000000
//        Final position: 43484.287140
//        sbsiewert@o251-01:~/public_html/csci551/code/hello_cluster$ mpiexec -n 3 ./chatgpttrain
//        Final velocity: 0.000000
//        Final position: 17341.429140
//        sbsiewert@o251-01:~/public_html/csci551/code/hello_cluster$
//

#include <stdlib.h> // ChatGPT goof - it forgot this define for use of "free"
#include <mpi.h>
#include <stdio.h>
#include "ex4.h"

// ChatGPT goofed here and did not define this
#define SAMPLE_RATE (1)

int main(int argc, char** argv) 
{

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes and rank of the current process
    int num_processes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ChatGPT goofed here
    // Read the acceleration data from the file
    //double* acceleration = NULL;
    double* acceleration = DefaultProfile;

    // ChatGPT goofed here
    //int num_samples = read_data(&acceleration);
    int num_samples = sizeof(DefaultProfile)/sizeof(double);

    // Calculate the number of samples each process will handle
    int samples_per_process = num_samples / num_processes;

    // Calculate the start and end indices of the data segment for the current process
    int start_index = rank * samples_per_process;
    int end_index = (rank == num_processes - 1) ? num_samples - 1 : start_index + samples_per_process - 1;

    // Calculate the velocity and position for the current process's data segment
    double velocity = 0.0;
    double position = 0.0;

    for (int i = start_index; i < end_index; i++) 
    {
        double interval = (double)(i + 1) / SAMPLE_RATE - (double)i / SAMPLE_RATE;
        double area = (acceleration[i] + acceleration[i + 1]) / 2 * interval;
        velocity += area;
        position += velocity * interval + area / 2;
    }

    // Send the final velocity and position from each process to the master process
    if (rank != 0) 
    {
        MPI_Send(&velocity, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&position, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else 
    {
        // Receive the final velocity and position from each process and add them up
        double total_velocity = velocity;
        double total_position = position;

        for (int i = 1; i < num_processes; i++) 
        {
            MPI_Recv(&velocity, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&position, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_velocity += velocity;
            total_position += position;
        }

        // Output the final velocity and position
        printf("Final velocity: %lf\n", total_velocity);
        printf("Final position: %lf\n", total_position);
    }

    // ChatGPT goofed here - can't free a static initializer!
    // Free the memory used for the acceleration data
    //free(acceleration);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
