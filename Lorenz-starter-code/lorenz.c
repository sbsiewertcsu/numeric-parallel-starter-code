#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define N 10000

// Simple implementation of forward integration of classic Lorenz equations to
// provide a simple example of complex data for visualization in 3D.
//
int main(int argc,char **argv)
{
   int i=0;
   double x0,y0,z0,x1,y1,z1;

   // Set the integration step size
   double h = 0.01;

   // Lorenz system parameters - see Wikipedia for more backround
   //
   // http://en.wikipedia.org/wiki/Lorenz_equations
   //
   // These are same values used originally by Edward Lorenz to model
   // atmospheric convection - a complex process.
   //
   double a = 10.0;
   double b = 28.0;
   double c = 8.0 / 3.0;

   // Set initial state here
   x0 = 0.1;
   y0 = 0;
   z0 = 0;

   // Use forward step integration at a small step size for N steps
   // to integrate the Lorenz equations in X, Y, Z space.
   //
   for (i=0; i<N; i++) 
   {
      // Lorenz equations - ordinary differential equation forward
      // integration.
      x1 = x0 + h * a * (y0 - x0);
      y1 = y0 + h * (x0 * (b - z0) - y0);
      z1 = z0 + h * (x0 * y0 - c * z0);

      // Current state equals newly computed state after integration
      x0 = x1;
      y0 = y1;
      z0 = z1;

      // Print out current state
      printf("%d, %g, %g, %g\n",i,x0,y0,z0);
   }
}
