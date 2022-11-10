// https://www.geeksforgeeks.org/program-for-bisection-method/
//
// C++ program for implementation of Bisection Method for 
// solving equations 
//
// Refactored to improve commenting and variable names and to set
// error tolerance to tolerance for digits of precision.
//
// Sam Siewert, 4/26/22
//
// For theory, start with Wikipedia - https://en.wikipedia.org/wiki/Bisection_method
//
// Draw some examples and convince yourself that this is better than just brute force based on searching
// for the ZERO in LEFT or RIGHT half of the original interval and repeating until we achieve a ZERO within
// error tolerance, which can be as small as the smallest number we can represent with our floating point precision.
//
#include<bits/stdc++.h> 
using namespace std; 

// Either define acceptable
//#define EPSILON 0.000000000000001
//
// OR
// Use float.h defined smallest EPSILON for digits of precision
//
#include <float.h>
#define EPSILON (DBL_EPSILON)
  
// An example function whose solution is determined using 
// Bisection Method. The function is x^3 - x^2  + 2 
double func(double x) 
{ 
    //return x*x*x - x*x + 2; 

    // As can be seen with Desmos, this example equation has roots near: -2.851, 4.758, and 7.792
    return (-(x*x*x) + 9.7*(x*x) -1.3*x -105.7);
} 
  
// Prints root of func(x) with error of EPSILON 
void bisection(double a, double b) 
{ 
    int count=0;

    // If the interval contains at least 1 zero crossing, then func(a) * func(b) will be less than 0.0
    //
    // Here we will return from the funciton if there are no zero crossings or an even number of multiple
    // crossings.
    //
    // the one potential pitfall is that the interval could contain multiple zero crossings, so it is best
    // to first analyze the function with graphing, or to search a small interval repeatedly.
    //
    if (func(a) * func(b) >= 0) 
    { 
        cout << "You have not assumed right a and b\n"; 
        return; 
    } 
  
    double c = a; 

    while ((b-a) >= EPSILON) 
    { 

        // Find middle point 
        //
        // This is the part that is like Zeno's paradox, we just cut the interval in half here
        c = (a+b)/2; 
  
        // Check if middle point is root  -- this is the orignal code, but seems sketchy to compare
        // to zero with floating point.  Improved to check to see if we in fact ever hit this or just
        // wind up exiting when the interval is less than EPSILON, but as long as it is ZERO within
        // floating point error, this does work.
        if (func(c) == 0.0) 
        {
            cout << "function at midpoint " << c << " is ZERO within " << EPSILON << " and f(c)=" << func(c) << endl;
            break; 
        }
  
        // Decide the side to repeat the steps 
        //
        // If the ZERO crossing is in the right half of the currrent interval, then make the 
        // new interval c to b, recalling that b > a, and c is in the middle
        else if (func(c)*func(a) < 0) 
            b = c; 

        // If the ZERO crossing is in the left half of the current interval, then make the
        // new interval a to c, recalling that a < b, and c is in the middle
        else
            a = c; 

        // It's all very similar to Zeno's paradox, but rather than halving the distance to a landmark, here
        // we cut the interval in half based on which side we detect the ZERO (landmark) - we can't see it
        // like Zeno, we detect the ZERO as f(c)*f(a) < 0 - crossing was between midpoint and left side, or
        // it must be in the other half
        count++;
    } 
    cout << "After " << count << " iterations,"<< " the value of root is : " << fixed << setprecision(15) << c << endl; 
} 


// Driver program to test above function 
int main() 
{ 
    // Initial values assumed - a large interval that contains 1 or more roots
    double a =-200, b = 300; 

    bisection(a, b); 
    return 0; 
} 

