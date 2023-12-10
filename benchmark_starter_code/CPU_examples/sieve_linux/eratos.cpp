#include <iostream>

using namespace std;


#define MAX (30ULL)
#define CODE_LENGTH ((sizeof(unsigned char))*8ULL)

unsigned char *isprime;

int chk_isprime(unsigned long long int i)
{
    unsigned long long int idx;
    unsigned int bitpos;

    idx = i/(CODE_LENGTH);
    bitpos = i % (CODE_LENGTH);

    //cout << "i=" << i << ", idx=" << idx << ", bitpos=" << bitpos << "\n";

    return(((isprime[idx]) & (1<<bitpos))>0);
}

int set_isprime(unsigned long long int i, unsigned char val)
{
    unsigned long long int idx;
    unsigned int bitpos;

    idx = i/(CODE_LENGTH);
    bitpos = i % (CODE_LENGTH);

    cout << "i=" << i << ", idx=" << idx << ", bitpos=" << bitpos << "\n";

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

    cout << "idx=" << (MAX/(CODE_LENGTH)) << "\n";

    for(idx=(MAX/(CODE_LENGTH)); idx >= 0; idx--)
    {
        cout << "idx=" << idx << ", " << isprime[idx] << "\n";
    }
    cout << "\n";

}


int main(void)
{
        unsigned long long int i, j;
        unsigned long long int p=2;
        unsigned int cnt=0;
        unsigned long long int thread_idx=0;
		int idx=0, ridx=0;

        cout << "max uint = " << (0xFFFFFFFF) << "\n";
        cout << "max long long = " << (0xFFFFFFFFFFFFFFFFULL) << "\n";

        if( !( ( isprime= new unsigned char[(((size_t)(MAX/(CODE_LENGTH))+1))] ) > 0) )
        {
            cout << "new failed\n"; 
            exit(-1);
        }

        // Not prime by definition
        // 0 & 1 not prime, 2 is prime, 3 is prime, assume others prime to start
        isprime[0]=0xFC; 
        for(i=2; i<MAX; i++) { set_isprime(i, 1); }
  
        for(i=0; i<MAX; i++) 
		{ 
			cout << "isprime=" << chk_isprime(i) << "\n"; 
		}

        print_isprime();

        while( (p*p) <=  MAX)
        {
            // cout << "p=" << p << "\n";

            // invalidate all multiples of lowest prime so far
            // 
            // simple to compose into a grid of invalidations
            //

            for(j=2*p; j<MAX+1; j+=p)
            {
                //cout << "j=" << j << "\n";
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
                // cout << "i=" << i << "\n"; 
            }
        }
        cout << "\nNumber of primes [0.." << MAX << "]=" << cnt << "\n";

        return idx;
}
