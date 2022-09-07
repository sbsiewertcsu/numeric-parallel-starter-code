struct divnum_struct
{
    int q;
    int rem;
};

static int ran_seed;
static int ran_a;
static int ran_m;
static int ran_q;
static int ran_r;


void reset_random(int seed)
{
    ran_seed = (seed) ? seed : 1;  /* if seed is 0, set it to 1 */
    ran_a = 16807;
    ran_m = 2147483647;
    ran_q = 127773;
    ran_r = 2836;
}


int divop(int m, int a, struct divnum_struct *divnump)
{
    unsigned int q=0;
    unsigned int rem = m;

#if 0
    int i=0;

    /* speed up by first shifting to divide by 2 each iteration first */
    q = 1;
    for ( ; rem > a; (q=q<<1) && (i < 32))
    {
        printf("q=%d, rem=%d\n", q, rem);
        rem = rem >> 1;
    }
#endif

    /* standard 2's complement implementation of div */
    for ( ; rem > a; q++)
    {
        rem = rem - a;
    }

    divnump->q = q;
    divnump->rem = rem;

    return(1);
}


int random(void)
{
    int test;
    struct divnum_struct divnum;

    divop(ran_seed, ran_q, &divnum);

    test = (ran_a * divnum.rem) - (ran_r * divnum.q);

    if (test > 0)
        ran_seed = test;
    else
        ran_seed = test + ran_m;

    //return(ran_seed); /* returns a number between 0 and (2^31)-1 */
    return(ran_seed >> 27);
}


char msgStr[]="IF YOU SOLVE THIS CHALLENGE STRING SEND ME AN EMAIL";
char cipStr[]="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
char newStr[]="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
#define ALPHA_SIZE 26 
#define ALPHA_BASE (int)('A')


int main(int argc, char *argv[])
{
  int i, offset, val;

  reset_random(1);

  // Encypher
  for(i=0;i<strlen(msgStr);i++)
  {
     offset=random()+1;
     //printf("offset=%d\n", offset);
     val = msgStr[i] + offset;
     if(val > 90) val = ALPHA_BASE + (val - 90);

     if(msgStr[i] == ' ')
        cipStr[i] = msgStr[i];
     else
        cipStr[i] = (char)val;
  }

  printf("cypher string = %s\n", cipStr);

  reset_random(1);

  //Decypher
  for(i=0;i<strlen(cipStr);i++)
  {
     offset=random()+1;
     //printf("offset=%d\n", offset);
     val = cipStr[i] - offset;
     if(val < 65) val = 90 - (ALPHA_BASE - val);
     if(cipStr[i] == ' ')
        newStr[i] = cipStr[i];
     else
        newStr[i] = (char)val;
  }

  printf("plain text string = %s\n", newStr);

}

