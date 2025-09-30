#include <stdio.h>

int main(void)
{
    // At limit of 7 digits that can be trusted
    float ex1_a=123456.5, ex1_b=123455.5;
    float ex2_a=123456.6, ex2_b=123455.4;
    double dex1_a=123456.5, dex1_b=123455.5;
    double dex2_a=123456.6, dex2_b=123455.4;

    // Beyond limit of 7 digits that can be trusted
    float ex3_a=1234567.5, ex3_b=1234566.5;
    float ex4_a=1234567.6, ex4_b=1234566.4;
    double dex3_a=1234567.5, dex3_b=1234566.5;
    double dex4_a=1234567.6, dex4_b=1234566.4;

    printf("FLOAT  Ex 1: a-b=%f, should be 1.0, for %f-%f, but %f, %f\n", (ex1_a-ex1_b), 123456.5, 123455.5, ex1_a, ex1_b);
    printf("FLOAT  Ex 2: a-b=%f, should be 1.2, for %f-%f, but %f, %f\n", (ex2_a-ex2_b), 123456.6, 123455.4, ex2_a, ex2_b);
    printf("FLOAT  Ex 3: a-b=%f, should be 1.0, for %f-%f, but %f, %f\n", (ex3_a-ex3_b), 1234567.5, 1234566.5, ex3_a, ex3_b);
    printf("FLOAT  Ex 4: a-b=%f, should be 1.2, for %f-%f, but %f, %f\n", (ex4_a-ex4_b), 1234567.6, 1234566.4, ex4_a, ex4_b);
    printf("\n");
    printf("DOUBLE Ex 1: a-b=%lf, should be 1.0, for %lf-%lf=%lf\n", (dex1_a-dex1_b), dex1_a, dex1_b, 1.0);
    printf("DOUBLE Ex 2: a-b=%lf, should be 1.2, for %lf-%lf=%lf\n", (dex2_a-dex2_b), dex2_a, dex2_b, 1.2);
    printf("DOUBLE Ex 3: a-b=%lf, should be 1.0, for %lf-%lf=%lf\n", (dex3_a-dex3_b), dex3_a, dex3_b, 1.0);
    printf("DOUBLE Ex 4: a-b=%lf, should be 1.2, for %lf-%lf=%lf\n", (dex4_a-dex4_b), dex4_a, dex4_b, 1.2);
}
