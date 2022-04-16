#include <stdio.h>

void main(void)
{
    float ex1_a=1234567.5, ex1_b=1234566.5;
    float ex2_a=1234567.6, ex2_b=1234566.4;

    double dex1_a=1234567.5, dex1_b=1234566.5;
    double dex2_a=1234567.6, dex2_b=1234566.4;

    printf("FLOAT  Ex 1: a-b=%f, should be 1.0, but %f-%f\n", (ex1_a-ex1_b), ex1_a, ex1_b);
    printf("FLOAT  Ex 2: a-b=%f, should be 1.2, but %f-%f\n", (ex2_a-ex2_b), ex2_a, ex2_b);

    printf("\n");

    printf("DOUBLE Ex 1: a-b=%f, should be 1.0 and %f-%f=1.0\n", (dex1_a-dex1_b), dex1_a, dex1_b);
    printf("DOUBLE Ex 2: a-b=%f, should be 1.2 and %f-%f=1.2\n", (dex2_a-dex2_b), dex2_a, dex2_b);
}
