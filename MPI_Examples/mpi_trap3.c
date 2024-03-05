/* File:     mpi_trap3.c
 * Purpose:  Use MPI to implement a parallel version of the trapezoidal 
 *           rule.  This version uses collective communications to 
 *           distribute the input data and compute the global sum.
 *
 * Input:    The endpoints of the interval of integration and the number
 *           of trapezoids
 * Output:   Estimate of the integral from a to b of f(x)
 *           using the trapezoidal rule and n trapezoids.
 *
 * Compile:  mpicc -g -Wall -o mpi_trap2 mpi_trap2.c
 * Run:      mpiexec -n <number of processes> ./mpi_trap2
 *
 * Algorithm:
 *    1.  Each process calculates "its" interval of
 *        integration.
 *    2.  Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3a. Each process != 0 sends its integral to 0.
 *    3b. Process 0 sums the calculations received from
 *        the individual processes and prints the result.
 *
 * Note:  f(x) is all hardwired.
 *
 * IPP:   Section 3.4.2 (pp. 104 and ff.)
 */
#include <stdio.h>
#include <math.h>

/* We'll be using MPI routines, definitions, etc. */
#include <mpi.h>

/* Get the input values */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p,
      int* n_p);

/* Calculate local integral  */
double Trap(double left_endpt, double right_endpt, int trap_count, 
   double base_len);    
double LeftRiemann(double left_endpt, double right_endpt, int rect_count, 
   double base_len);    

/* Function we're integrating */
double ex3_accel(double time);
double ex3_vel(double time);
double ex3_pos(double time);
double funct_to_integrate(double x); 

int main(void) {
   int my_rank, comm_sz, n, local_n;   
   double a, b, step_size, local_a, local_b;
   double local_int_area, total_int_area;

   /* Let the system do what it needs to start up MPI */
   MPI_Init(NULL, NULL);

   /* Get my process rank */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* Find out how many processes are being used */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   Get_input(my_rank, comm_sz, &a, &b, &n);

   if(my_rank == 0) printf("my_rank=%d, a=%15.14lf, b=%15.14lf, number of total steps=%d\n", my_rank, a, b, n);

   step_size = (b-a)/n;  /* step is the same for all processes */
   local_n = n/comm_sz;  /* So is the number of quadratures  */

   /* Length of each process' interval of
    * integration = local_n*step_size.  So my interval
    * starts at: */
   local_a = a + my_rank*local_n*step_size;
   local_b = local_a + local_n*step_size;

   printf("my_rank=%d, start a=%lf, end b=%lf, number of quadratures = %d, step_size=%lf\n",
           my_rank, local_a, local_b, local_n, step_size);

   local_int_area = Trap(local_a, local_b, local_n, step_size);
   //local_int_area = LeftRiemann(local_a, local_b, local_n, step_size);

   printf("After LeftRiemann: my_rank=%d, integrated area = %15.14lf, step_size %lf, number quadratures=%d\n", 
           my_rank, local_int_area, step_size, local_n);

   /* Add up the integrals calculated by each process */
   MPI_Reduce(&local_int_area, &total_int_area, 1, MPI_DOUBLE, MPI_SUM, 0,
         MPI_COMM_WORLD);

   /* Print the result */
   if (my_rank == 0) {
      printf("After Reduce: with n = %d quadratures, our estimate\n", n);
      printf("of the integral from %lf to %lf = %15.14lf\n",
          a, b, total_int_area);
   }

   /* Shut down MPI */
   MPI_Finalize();

   return 0;
} /*  main  */

/*------------------------------------------------------------------
 * Function:     Get_input
 * Purpose:      Get the user input:  the left and right endpoints
 *               and the number of quadratures
 * Input args:   my_rank:  process rank in MPI_COMM_WORLD
 *               comm_sz:  number of processes in MPI_COMM_WORLD
 * Output args:  a_p:  pointer to left endpoint               
 *               b_p:  pointer to right endpoint               
 *               n_p:  pointer to number of quadratures
 */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p,
      int* n_p) {
   int rc=0;

   if (my_rank == 0) {
      printf("Enter a, b, and n\n");
      rc=scanf("%lf %lf %d", a_p, b_p, n_p); if(rc < 0) perror("Get_input");
   } 
   MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
}  /* Get_input */

/*------------------------------------------------------------------
 * Function:     Trap
 * Purpose:      Serial function for estimating a definite integral 
 *               using the trapezoidal rule
 * Input args:   left_endpt
 *               right_endpt
 *               trap_count 
 *               base_len
 * Return val:   Trapezoidal rule estimate of integral from
 *               left_endpt to right_endpt using trap_count
 *               trapezoids
 */
double Trap(
      double left_endpt  /* in */, 
      double right_endpt /* in */, 
      int    trap_count  /* in */, 
      double base_len    /* in */) {
   double estimate, x; 
   int i;

   estimate = (funct_to_integrate(left_endpt) + funct_to_integrate(right_endpt))/2.0;

   for (i = 1; i <= trap_count-1; i++) 
   {
      x = left_endpt + i*base_len;
      estimate += funct_to_integrate(x);
   }
   estimate = estimate*base_len;

   return estimate;
} /*  Trap  */


double LeftRiemann(
      double left_endpt, 
      double right_endpt, 
      int    rect_count, 
      double base_len) 
{
   double left_value, x, area=0.0; 
   int i;

   // estimate of function on left side to forward integrate
   x = left_endpt;
   left_value = funct_to_integrate(x);

   for (i = 1; i <= rect_count; i++) 
   {
      // add area of each rectangle to overall area sum
      area += left_value * base_len;

      // advance x by base length for new values to add to area
      x += base_len;
      left_value = funct_to_integrate(x);
   }

   return area;

} /*  LeftRiemann  */



/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double funct_to_integrate(double x) 
{
    //return sin(x);
    //return(ex3_accel(x));
    //return(ex3_vel(x));
    //return(ex3_pos(x));
    return 10.0;
}


double ex3_accel(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that acceleration will peak to result in translation of 122,000.0 meters
    //static double ascale=0.2365893166123;
    static double ascale=0.236589076381454;

    return (sin(time/tscale)*ascale);
}


// determined based on known anti-derivative of ex4_accel function
double ex3_vel(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static double vscale=0.236589076381454*1800.0/(2.0*M_PI);

    return ((-cos(time/tscale)+1)*vscale);
}


// determined based on known anti-derivative of ex4_vel function
double ex3_pos(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static double vscale=0.236589076381454*1800.0/(2.0*M_PI);

    return ((-tscale*(sin(time/tscale)+time))*vscale);
}
