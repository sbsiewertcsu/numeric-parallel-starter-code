#include <iostream>
#include <stdlib.h>

using namespace std;

#define MAX (30ULL)
#define TRUE (1)
#define FALSE (0)

typedef struct intlist_s
{
	unsigned int candidate;
	int isprime;
} intlist_t;


void crossOutNonPrimes(unsigned int p, intlist_t *intlist)
{
	unsigned int idx;

	cout << "crossOutNonPrimes for " << p << endl;

	// Starting from 2p, count up in increments of p and mark each of these 
	// numbers greater than p itself in the list
	for(idx=p+p; idx < MAX; idx=idx+p) 
	{
		intlist[idx].isprime = FALSE;
	}

	// Find the first number greater than p in the list that is not marked.
	// If there was no such number, stop.
	for(idx=p+1; idx < MAX; idx++)
	{
		if(intlist[idx].isprime == TRUE)
		{
			p = intlist[idx].candidate;
			crossOutNonPrimes(p, intlist);
		}
	}

	return;
}


int main(void)
{
	unsigned int p=2, idx=0;  // Initially, let p equal 2, the first prime number.

	// Create list consecutive integers from 2 to MAX: (2, 3, 4, ..., MAX).	
	intlist_t *intlist;

	if( !( ( intlist = new intlist_t [MAX] ) > 0) )
    {
		cout << "new failed\n"; 
		exit(-1);
    }
	else
	{
		cout << "Integer list allocated\n";
	}

	// Initialize this list and mark 0, 1 and 2 as primes by definition and assume others prime until crossed off
	for(idx=0; idx < MAX; idx++)
	{
		intlist[idx].candidate=idx;
		intlist[idx].isprime = TRUE; // assumed prime until proven otherwise, set false if it would be marked
	}

	cout << "Integer list initialized\n";
	
	crossOutNonPrimes(p, intlist);

	for(idx=0; idx < MAX; idx++)
	{
		if(intlist[idx].isprime == TRUE)
			cout << intlist[idx].candidate << ", ";
	}

	cout << "...\n";

	return 1;
}

