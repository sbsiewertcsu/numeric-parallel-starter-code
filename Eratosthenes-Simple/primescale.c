#include <stdio.h>
#include <stdlib.h>


// Simple code to implement the original Eratosthenes Sieve
//
// Has been tested up to 1 billion using - https://primes.utm.edu/howmany.html
// to verify the expected # of primes in this range and each prime can be printed
// and compared with this list - http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
// Another good source to verify - https://en.wikipedia.org/wiki/Prime-counting_function
// Yet another - https://t5k.org/howmany.html
//
// Should work as long as you have sufficent memory to malloc the bitmap.
//
#define MAX (4294967295ULL)
#define DEMO (1000000ULL)
#define CODE_LENGTH ((sizeof(unsigned char))*8ULL)

// Static declaration replaced by malloc
//
//unsigned char isprime[(MAX/(CODE_LENGTH))+1];
unsigned char *isprime;

// List of the primes - assumes that at most 10% of numbers are prime
#define MAX_PRIMES (500000000)
unsigned int primelist[MAX_PRIMES];

int chk_isprime(unsigned long long int i)
{
    unsigned long long int idx;
    unsigned int bitpos;

    idx = i/(CODE_LENGTH);
    bitpos = i % (CODE_LENGTH);

    //printf("i=%llu, idx=%llu, bitpos=%u\n", i, idx, bitpos);

    return(((isprime[idx]) & (1<<bitpos))>0);
}

// Be careful wtih thread safety for set_isprime. You can get lucky, but
// shared memory read, modify, write is in general unsafe.
//
// You can consider:
//
// 1) Not threading this one function - although it is one for which speed-up benefits are high
// 2) Mapping threads to specific ranges so they do not index common locations (this is a chance to be
//    creative with map and reduce). Look at cross out carefully and imagine launching threads from different
//    starter primes to cross out multiples - note that crossing out the same non-prime is not a problem. The
//    problem is reading a non-prime and treating it as prime because it should be crossed out, but has not been
//    crossed out yet. This is a strategy for thread safety using thread indexing methods to avoid RMW or globally shared
//    memory.
// 3) Using atomic
// 4) Using MUTEX omp critical - usually not work it, see option #1
// 5) Coming up with a stack-based approach using map and reduce
//
int set_isprime(unsigned long long int i, unsigned char val)
{
    unsigned long long int idx;
    unsigned int bitpos;

    idx = i/(CODE_LENGTH);
    bitpos = i % (CODE_LENGTH);

    //printf("i=%llu, idx=%llu, bitpos=%u\n", i, idx, bitpos);

    if(val > 0)
    {
        isprime[idx] = isprime[idx] | (1<<bitpos);
    }
    else
    {
        isprime[idx] = isprime[idx] & (~(1<<bitpos));
    }

	return bitpos;
}


void print_isprime(void)
{
    long long int idx=0;

    printf("idx=%lld\n", (MAX/(CODE_LENGTH)));

    for(idx=(MAX/(CODE_LENGTH)); idx >= 0; idx--)
    {
        printf("idx=%lld, %02X\n", idx, isprime[idx]);
    }
    printf("\n");

}


int main(int argc, char *argv[])
{
    int thread_count=8;
    unsigned long long int i, j;
    unsigned long long int p=2;
    unsigned int cnt=0;
    unsigned int list_cnt=0;
    unsigned long long int thread_idx=0;
    unsigned long long int sp=0, max_n=DEMO;
	int idx=0, ridx=0, primechk;

    if (argc == 2)
    {
      sscanf(argv[1], "%d", &thread_count);
      max_n=DEMO;
      printf ( "Using: %d threads for %llu range\n", thread_count, max_n);
    }
    else if (argc == 3)
    {
      sscanf(argv[1], "%d", &thread_count);
      sscanf(argv[2], "%llu", &max_n);
      printf ( "Using: %d threads for %llu range\n", thread_count, max_n);
    }
    else
    {
      printf ( "usage: primescale <number threads> <max n>\n");
      thread_count=8;
      max_n=DEMO;
      printf ( "Using defaults: %d threads for %llu range\n", thread_count, max_n);
    }

    printf("max uint = %u\n", (0xFFFFFFFF));
    printf("max long long = %llu\n", (0xFFFFFFFFFFFFFFFFULL));

    if(!((isprime=malloc((size_t)(max_n/(CODE_LENGTH))+1)) > 0))
    {
        perror("malloc");
        printf("insufficient memory for prime sieve up to %lld\n", max_n);
        exit(-1);
    }
    else
    {
        printf("Sufficient memory for prime sieve up to %lld using %llu Mbytes at address=%p\n", 
               max_n, ((max_n/(CODE_LENGTH))+1)/(1024*1024), isprime);
    }

    // Not prime by definition
    // 0 & 1 not prime, 2 is prime, 3 is prime, assume others prime to start
    isprime[0]=0xFC; 

// Be careful with threading set_isprime and thread safety for global
// cross-out array of bits.
//
#pragma omp parallel for num_threads(thread_count)
    for(i=2; i<max_n; i++) 
    {
        set_isprime(i, 1); 
    }
  
#pragma omp parallel for num_threads(thread_count)
    for(i=0; i<max_n; i++) 
    { 
        primechk = chk_isprime(i);
        //printf("isprime=%d\n", primechk); 
    }

    // will all be TRUE here or 0xFF for all except 0 & 1 so  0xFC for index=0
    //
    // for all other number bit array locations, we start out assuming they are prime,
    // so bit=1, and as we mark them non-prime, we flip that bit to bit=0
    //print_isprime();


    while( (p*p) <=  max_n)
    {
        //printf("p=%llu\n", p);

        // invalidate all multiples of lowest prime so far
        // 
        // simple to compose into a grid of invalidations
        //

#pragma omp parallel for num_threads(thread_count)
        for(j=2*p; j<max_n+1; j+=p)
        {
            //printf("j=%llu\n", j);
            set_isprime(j,0);
        }

        // find next lowest prime - sequential process
        for(j=p+1; j<max_n+1; j++)
        {
            if(chk_isprime(j)) 
            { 
                p=j; 
                break;  // issue for speed-up with OpenMP, pragma won't work as coded
            }
        }

    }


// Note that a reduction on the global cnt variable is necessary for correct results
//
// This works because addition is commutative.
// 
// A quick review of math properties:
// 1) Commutative - https://en.wikipedia.org/wiki/Commutative_property, means we can add up
//    cnt that each thread has a copy of in any order we want by reducing at the end
// 2) Associative - https://en.wikipedia.org/wiki/Associative_property has to do with order
//    of the application of operators, which can be disambiguated using parenthesis
// 3) Distributive - https://en.wikipedia.org/wiki/Distributive_property which has to do with
//    multiplying a number by a grouped sum, whereby 3x(1+2)=3+6=9 for example.
//
// In parallel programming, we often make use of the commutative property to "reduce" results
// where partial results are determined by each thread and must be merged into a single result.
//
// Examples similar to summing are MAX and MIN. Reduction only works for operations that are in
// fact commutative (can be applied to results arranged in any order).  If this is not true, the
// results may be corrupted and we might need to use omp critical to serialize.
//
cnt=0;
#pragma omp parallel for num_threads(thread_count) reduction(+:cnt)
    for(i=0; i<max_n+1; i++)
    {
        if(chk_isprime(i))
        { 
            cnt++; 
            //printf("i=%llu\n", i); 
        }
    }

    // Can't thread this as the cnt is global and indexes the list of primes
    // unless we use omp critical, but that would serialize
    // for demonstration we just do the loop again with no threading and save off
    // each prime.
    //
    // For time comparison, comment this out, but for use with factoring a large
    // SP = P1 x P2, the list of primes is necessary for search.
    //
    for(i=0; i<max_n+1; i++)
    {
        if(chk_isprime(i))
        { 
            primelist[list_cnt]=i;
            list_cnt++; 
            //printf("i=%llu\n", i); 
        }
    }

    printf("\nNumber of MAX primes [0..%llu]=%u\n\n", max_n, cnt);

    // Let's now compute an example large semi-prime
    sp = ((unsigned long long)primelist[cnt-(cnt/2)]) * ((unsigned long long)primelist[cnt-(cnt/8)]);

    printf("Example large SP is %llu, factored into p1=%u, p2=%u\n", 
           sp, primelist[cnt-(cnt/2)], primelist[cnt-(cnt/8)]);

    printf("Now we could use the primelist and search for the first prime where (sp mod p1) == 0\n");
    printf("Once we find the first zero modulo, then p2 = sp / p1 and we have our factors!\n");

    return (i);
}

