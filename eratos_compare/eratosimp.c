#include <stdio.h>


//#define MAX (100000000)
#define MAX (1200000000)
//#define MAX (0xFFFFFFFF)
unsigned char isprime[MAX+1];
unsigned int value[MAX+1];

void main(void)
{
    int i, j;
    unsigned int p=2, cnt=0;

    // not prime by definition
    isprime[0]=0; value[0]=0;
    isprime[1]=0; value[1]=1;

    for(i=2; i<MAX+1; i++) { isprime[i]=1; value[i]=i; }

    while( (p*p) <=  MAX)
    {
        // invalidate all multiples of lowest prime so far
        for(j=2*p; j<MAX+1; j+=p) isprime[j]=0;

        // find next lowest prime
        for(j=p+1; j<MAX+1; j++) { if(isprime[j]) { p=j; break; } }
    }

    for(i=0; i<MAX+1; i++) { if(isprime[i]) { cnt++; } }
    printf("\nNumber of primes [0..%d]=%u\n\n", MAX, cnt);
}
