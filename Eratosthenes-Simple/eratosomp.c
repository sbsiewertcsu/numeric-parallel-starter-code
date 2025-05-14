#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// Simple code to implement the original Eratosthenes Sieve
//
// Has been tested up to 1 billion using - https://primes.utm.edu/howmany.html
// to verify the expected # of primes in this range and each prime can be printed
// and compared with this list - http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
//
// Should work as long as you have sufficent memory to malloc the bitmap.
//
// define GIANT for test up to 4 billion, else will do range to 1 million
//
#define GIANT

#ifdef GIANT
#define MAX (4000000000ULL) // limit is 4 billion based on 32-bit primes, 64-bit SP
#else
#define MAX (1000000ULL)
#endif

#define CODE_LENGTH ((sizeof(unsigned char))*8ULL)

// Static declaration replaced by malloc
//
//unsigned char isprime[(MAX/(CODE_LENGTH))+1];
unsigned char *isprime;

// List of the primes - assumes that at most 10% of numbers are prime
// E.g., 400 million primes or less in the range 0...4 billion
#define MAX_PRIMES (400000000)
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
    unsigned long long int sp=0;
	int idx=0, ridx=0, primechk;

    // Instrumented time-stamps in code
    struct timespec now, start;
    double fnow=0.0, fstart=0.0, faccum=0.0;

    if(argc == 2)
    {
        sscanf(argv[1], "%d", &thread_count);
        printf("Setting thread_count to %d\n", thread_count);
    }
    else
    {
        printf("Using DEFAULT thread_count of %d\n", thread_count);
    }

    printf("max uint = %u\n", (0xFFFFFFFF));
    printf("max long long = %llu\n", (0xFFFFFFFFFFFFFFFFULL));

    if(!((isprime=malloc((size_t)(MAX/(CODE_LENGTH))+1)) > 0))
    {
        perror("malloc");
        printf("insufficient memory for prime sieve up to %lld\n", MAX);
        exit(-1);
    }
    else
    {
        printf("Sufficient memory for prime sieve up to %lld using %llu Mbytes at address=%p\n", 
               MAX, ((MAX/(CODE_LENGTH))+1)/(1024*1024), isprime);
    }

    // Not prime by definition
    // 0 & 1 not prime, 2 is prime, 3 is prime, assume others prime to start
    isprime[0]=0xFC; 

// Be careful with threading set_isprime and thread safety for global
// cross-out array of bits.
//
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for num_threads(thread_count)
    for(i=2; i<MAX; i++) 
    {
        set_isprime(i, 1); 
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    fstart = (double)start.tv_sec  + (double)start.tv_nsec / 1000000000.0;
    fnow = (double)now.tv_sec  + (double)now.tv_nsec / 1000000000.0;
    printf("Initial set_isprime(%llu) in %lf secs\n", MAX, (fnow-fstart));
  
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for num_threads(thread_count)
    for(i=0; i<MAX; i++) 
    { 
        primechk = chk_isprime(i);
        //printf("isprime=%d\n", primechk); 
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    fstart = (double)start.tv_sec  + (double)start.tv_nsec / 1000000000.0;
    fnow = (double)now.tv_sec  + (double)now.tv_nsec / 1000000000.0;
    printf("chk_isprime(%llu) in %lf secs\n", MAX, (fnow-fstart));
  

    // will all be TRUE here or 0xFF for all except 0 & 1 so  0xFC for index=0
    //
    // for all other number bit array locations, we start out assuming they are prime,
    // so bit=1, and as we mark them non-prime, we flip that bit to bit=0
    //print_isprime();


    printf("Entering cross out loop\n"); faccum=0.0;

    while( (p*p) <=  MAX)
    {
        //printf("p=%llu\n", p);

        // invalidate all multiples of lowest prime so far
        // 
        // simple to compose into a grid of invalidations
        //

        clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for num_threads(thread_count)
        for(j=2*p; j<MAX+1; j+=p)
        {
            //printf("j=%llu\n", j);
            set_isprime(j,0);
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        fstart = (double)start.tv_sec  + (double)start.tv_nsec / 1000000000.0;
        fnow = (double)now.tv_sec  + (double)now.tv_nsec / 1000000000.0;
        faccum += (fnow-fstart);
        //printf("Cross out set_isprime(%llu) in %lf secs\n", MAX, (fnow-fstart));
  

        // find next lowest prime - sequential process
        for(j=p+1; j<MAX+1; j++)
        {
            if(chk_isprime(j)) 
            { 
                p=j; 
                break;  // issue for speed-up with OpenMP, pragma won't work as coded
            }
        }

    }
    printf("Done with cross out loop with set_isprime(%llu) time=%lf secs\n", MAX, faccum);


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
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for num_threads(thread_count) reduction(+:cnt)
    for(i=0; i<MAX+1; i++)
    {
        if(chk_isprime(i))
        { 
            cnt++; 
            //printf("i=%llu\n", i); 
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    fstart = (double)start.tv_sec  + (double)start.tv_nsec / 1000000000.0;
    fnow = (double)now.tv_sec  + (double)now.tv_nsec / 1000000000.0;
    printf("Total number of primes chk_isprime(%llu) in %lf secs", MAX, (fnow-fstart));
  

    // Can't thread this as the cnt is global and indexes the list of primes
    // unless we use omp critical, but that would serialize
    // for demonstration we just do the loop again with no threading and save off
    // each prime.
    //
    // For time comparison, comment this out, but for use with factoring a large
    // SP = P1 x P2, the list of primes is necessary for search.
    //
    for(i=0; i<MAX+1; i++)
    {
        if(chk_isprime(i))
        { 
            primelist[list_cnt]=i;
            list_cnt++; 
            //printf("i=%llu\n", i); 
        }
    }


    printf("\nNumber of primes [0..%llu]=%u\n\n", MAX, cnt);

    printf("List of primes (skipping by millions) is:\n");
    for(i=0; i<cnt+1; i++)
        if((i % 1000000) == 0) printf("%u\n", primelist[i]);

    // Let's now compute an example large semi-prime
    //
#ifdef GIANT
    sp = ((unsigned long long)primelist[1000000]) * ((unsigned long long)primelist[50000000]);

    printf("Example large SP is %llu, factored into p1=%u, p2=%u\n", 
           sp, primelist[1000000], primelist[50000000]);
#else
    sp = ((unsigned long long)primelist[10000]) * ((unsigned long long)primelist[50000]);

    printf("Example large SP is %llu, factored into p1=%u, p2=%u\n", 
           sp, primelist[10000], primelist[50000]);
#endif

    printf("Now we could use the primelist and search for the first prime where (sp mod p1) == 0\n");
    printf("Once we find the first zero modulo, then p2 = sp / p1 and we have our factors!\n");

    return (i);
}

