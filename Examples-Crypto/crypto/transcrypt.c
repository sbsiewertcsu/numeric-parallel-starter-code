#include <stdio.h>
#include <ctype.h>
#include <string.h>

#define BLOCK_SIZE (10)

//                          0  1  2  3  4  5  6  7  8  9
int transmap[BLOCK_SIZE]=  {6, 2, 7, 5, 1, 3, 4, 9, 0, 8};
int detransmap[BLOCK_SIZE]={8, 4, 1, 5, 6, 3, 0, 2, 9, 7};

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



char teststr[]="0123456789ABCD";

#define MAX_LENGTH (120)

void main(void)
{
    int i;
    size_t linelen;
    char *line=NULL;
    char depermuted_output[MAX_LENGTH];
    char permuted_output[MAX_LENGTH];

    printf("%s", teststr);
    printf("\n");

    printtrans(teststr, permuted_output, strlen(teststr));
    printf("\n");

    do
    {    
        printf("\nTRAN>");
        getline(&line, &linelen, stdin);
        printf("     ");
        printtrans(line, permuted_output, strlen(line));
        printf("     ");
        printdetrans(permuted_output, depermuted_output, strlen(permuted_output));

    }
    while(strncmp(line, "exit", 4) != 0);

    printf("\n");

}
