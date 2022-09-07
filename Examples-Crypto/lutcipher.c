#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>

#define ALPHABET_LENGTH (26)

char monoalphasub[]="QWERTYUIOPASDFGHJKLZXCVBNM"; 


int main(int argc, char *argv[])
{
    char basechar='A';
    char *newstring;
    int idx, lutidx, stridx;

    if(!(argc >= 2))
    {
        printf("usage: lutcypher <string-1> <string-2> ... <string-n>\n");
        exit(-1);
    }
    else
    {
        printf("will translate %s with base character %c=%d\n", 
                argv[1], basechar, (int)basechar);
    }


    for(stridx=1; stridx < argc; stridx++)
    {
        newstring = (char *)malloc(sizeof(char)*(strlen(argv[stridx])));

        for(idx=0; idx < strlen(argv[stridx]); idx++)
        {
            lutidx = argv[stridx][idx]-basechar;

            if(lutidx < ALPHABET_LENGTH)
            {
                printf("substitution index for %c is sub[%d]=%c\n", 
                       argv[stridx][idx], lutidx, monoalphasub[lutidx]);
                newstring[idx]=monoalphasub[lutidx];
            }
            else
            {
                printf("no substitution for %c, so using *\n", argv[stridx][idx]);
                newstring[idx]='*';
            }
        }

        printf("translation = %s\n", newstring);

        free((void *)newstring);
    }
} 
