//Code is taken from https://www.geeksforgeeks.org/nearest-prime-less-given-number-n/
//Authored by:  Shashank Mishra


// C++ program for the above approach
 
#include <bits/stdc++.h>
#include <time.h>
using namespace std;
 
// Function to return nearest prime number
int prime(int n)
{
 
    // All prime numbers are odd except two
    if (n & 1)
        n -= 2;
    else
        n--;
 
    int i, j;
    for (i = n; i >= 2; i -= 2) {
        if (i % 2 == 0)
            continue;
        for (j = 3; j <= sqrt(i); j += 2) {
            if (i % j == 0)
                break;
        }
        if (j > sqrt(i))
            return i;
    }
 
    // It will only be executed when n is 3
    return 2;
}
 
// Driver Code
int main()
{
    struct timespec now, start;
    double fnow, fstart;

    clock_gettime(CLOCK_MONOTONIC, &start);
    fstart = (double)start.tv_sec  + (double)start.tv_nsec / 1000000000.0;

    int n = 1000000000;
    cout << prime(n) << endl;

    clock_gettime(CLOCK_MONOTONIC, &now);
    fnow = (double)now.tv_sec  + (double)now.tv_nsec / 1000000000.0;
    printf("%lf sec elapsed\n", (fnow-fstart));
    return 0;
}
