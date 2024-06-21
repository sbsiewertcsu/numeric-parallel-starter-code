#include <stdio.h>
#include <ctype.h>
#include <string.h>

#define ALPHABET (26)

struct charmap
{
    char alpha;
    char beta;
};

struct charmap submap[ALPHABET] =
{
    {'A','N'}, {'B','A'}, {'C','S'}, {'D','I'}, {'E','J'}, {'F','K'}, {'G','C'},
    {'H','M'}, {'I','Q'}, {'J','R'}, {'K','F'}, {'L','B'}, {'M','D'}, {'N','G'},
    {'O','H'}, {'P','E'}, {'Q','L'}, {'R','O'}, {'S','P'}, {'T','W'}, {'U','T'},
    {'V','Z'}, {'W','Y'}, {'X','V'}, {'Y','U'}, {'Z','X'}
};

char teststr[]="TRANSLATE THIS!";
char inputstr[80];


// This is a slow iterative look-up table that could be
// optimized.

char findbeta(char alpha)
{
    int i;

    for(i=0; i < ALPHABET; i++)
    {
        if(submap[i].alpha == alpha)
            return(submap[i].beta);
    }

    return alpha;
}


char findalpha(char beta)
{
    int i;

    for(i=0; i < ALPHABET; i++)
    {
        if(submap[i].beta == beta)
            return(submap[i].alpha);
    }

    return beta;
}


void printbeta(char *input, char *output, int length)
{
    int i;

    for(i=0; i < length; i++)
    {
        output[i]=findbeta(toupper(input[i]));
        printf("%c", output[i]);
    }
    output[i]='\0';

    //printf("\ni=%d\n", i);
}


void printalpha(char *input, char *output, int length)
{
    int i;

    for(i=0; i < length; i++)
    {
        output[i]=findalpha(toupper(input[i]));
        printf("%c", output[i]);
    }
    output[i]='\0';

    //printf("\ni=%d\n", i);
}




#define MAX_LENGTH (120)

void main(void)
{
    int i;
    size_t linelen;
    char *line=NULL;
    char alpha_output[MAX_LENGTH];
    char beta_output[MAX_LENGTH];

    for(i=0; i < ALPHABET; i++)
        printf("%c ", submap[i].alpha);
    printf("\n");

    for(i=0; i < ALPHABET; i++)
        printf("%c ", submap[i].beta);
    printf("\n");

    printf("%s", teststr);
    printf("\n");

    printbeta(teststr, beta_output, strlen(teststr));
    printf("\n");

    do
    {    
        printf("\nBETA>");
        getline(&line, &linelen, stdin);
        printf("     ");
        printbeta(line, beta_output, strlen(line));
        printf("     ");
        printalpha(beta_output, alpha_output, strlen(beta_output));
    }
    while(strncmp(line, "exit", 4) != 0);

    printf("\n");
}
