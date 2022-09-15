#include <stdio.h>
#include <stdlib.h>

#define MAX (30ULL)
#define CODE_LENGTH ((sizeof(unsigned char))*8ULL)

//unsigned char isprime[(MAX/(CODE_LENGTH))+1];
unsigned char *isprime;

int chk_isprime(unsigned long long int i)
{
    unsigned long long int idx;
    unsigned int bitpos;

    idx = i/(CODE_LENGTH);
    bitpos = i % (CODE_LENGTH);

    //printf("i=%llu, idx=%llu, bitpos=%u\n", i, idx, bitpos);

    return(((isprime[idx]) & (1<<bitpos))>0);
}

int set_isprime(unsigned long long int i, unsigned char val)
{
    unsigned long long int idx;
    unsigned int bitpos;

    idx = i/(CODE_LENGTH);
    bitpos = i % (CODE_LENGTH);

    printf("i=%llu, idx=%llu, bitpos=%u\n", i, idx, bitpos);

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


int main(void)
{
        unsigned long long int i, j;
        unsigned long long int p=2;
        unsigned int cnt=0;
        unsigned long long int thread_idx=0;
		int idx=0, ridx=0;

        printf("max uint = %u\n", (0xFFFFFFFF));
        printf("max long long = %llu\n", (0xFFFFFFFFFFFFFFFFULL));

        if(!((isprime=malloc((size_t)(MAX/(CODE_LENGTH))+1)) > 0))
        {
            perror("malloc");
            exit(-1);
        }

        // Not prime by definition
        // 0 & 1 not prime, 2 is prime, 3 is prime, assume others prime to start
        isprime[0]=0xFC; 
        for(i=2; i<MAX; i++) { set_isprime(i, 1); }
  
        for(i=0; i<MAX; i++) { printf("isprime=%d\n", chk_isprime(i)); }
        print_isprime();

        while( (p*p) <=  MAX)
        {
            //printf("p=%llu\n", p);

            // invalidate all multiples of lowest prime so far
            // 
            // simple to compose into a grid of invalidations
            //

            for(j=2*p; j<MAX+1; j+=p)
            {
                //printf("j=%llu\n", j);
                set_isprime(j,0);
            }

            // find next lowest prime - sequential process
            for(j=p+1; j<MAX+1; j++)
            {
                if(chk_isprime(j)) { p=j; break; }
            }

        }


        for(i=0; i<MAX+1; i++)
        {
            if(chk_isprime(i))
            { 
                cnt++; 
                //printf("i=%llu\n", i); 
            }
        }
        printf("\nNumber of primes [0..%llu]=%u\n\n", MAX, cnt);

        return (i);
}