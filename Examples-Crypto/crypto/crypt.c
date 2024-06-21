#include <stdio.h>
#include <ctype.h>
#include <string.h>

#define BLOCK_SIZE (10)

//                          0  1  2  3  4  5  6  7  8  9
int transmap[BLOCK_SIZE]=  {6, 2, 7, 5, 1, 3, 4, 9, 0, 8};
int detransmap[BLOCK_SIZE]={8, 4, 1, 5, 6, 3, 0, 2, 9, 7};

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

void detranspose(char *input, char *output, int length)
{
    int i;
    char temp;

    for(i=0; i < length; i++)
    {
        output[i] = input[detransmap[i]];
    }
}


void transpose(char *input, char *output, int length)
{
    int i;
    char temp;

    for(i=0; i < length; i++)
    {
        output[i] = input[transmap[i]];
    }
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


void printtrans(char *input, char *permuted, int length)
{
    int idx=0;

    do
    {
        if((length-idx) < BLOCK_SIZE)
        {
            //printf("length=%d, idx=%d, %s\n", length, idx, &input[idx]);
            strncpy(&permuted[idx], &input[idx], (length-idx));
            idx = length;
        }
        else if((length-idx) >= BLOCK_SIZE)
        {
            //printf("length=%d, idx=%d, %s\n", length, idx, &input[idx]);
            transpose(&input[idx], &permuted[idx], BLOCK_SIZE);
            idx += BLOCK_SIZE;
        }
    }
    while(idx < length);

    permuted[idx]='\0';
    printf("%s", permuted);
}

void printdetrans(char *input, char *depermuted, int length)
{
    int idx=0;

    do
    {
        if((length-idx) < BLOCK_SIZE)
        {
            //printf("length=%d, idx=%d, %s\n", length, idx, &input[idx]);
            strncpy(&depermuted[idx], &input[idx], (length-idx));
            idx = length;
        }
        else if((length-idx) >= BLOCK_SIZE)
        {
            //printf("length=%d, idx=%d, %s\n", length, idx, &input[idx]);
            detranspose(&input[idx], &depermuted[idx], BLOCK_SIZE);
            idx += BLOCK_SIZE;
        }
    }
    while(idx < length);

    depermuted[idx]='\0';
    printf("%s", depermuted);
}



#define MAX_LENGTH (120)

void main(void)
{
    int i;
    size_t linelen;
    char *line=NULL;
    char alpha_output[MAX_LENGTH];
    char beta_output[MAX_LENGTH];
    char depermuted_output[MAX_LENGTH];
    char permuted_output[MAX_LENGTH];


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

    printtrans(beta_output, permuted_output, strlen(beta_output));
    printf("\n");

    do
    {
        printf("\nCRYPT>");
        getline(&line, &linelen, stdin);
        printf("     ");
        printbeta(line, beta_output, strlen(line));
        printf("     ");
        printtrans(beta_output, permuted_output, strlen(beta_output));
    }
    while(strncmp(line, "exit", 4) != 0);

    printf("\n");
}

