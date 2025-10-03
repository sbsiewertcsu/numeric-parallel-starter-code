#include <stdio.h>

int main(void)
{
    // Within limit of 7 digits that can be trusted
    float ex0_a=123.1, ex0_b=123.0;

    // At limit of 7 digits that can be trusted
    float ex1_a=123456.1, ex1_b=123456.0;
    float ex2_a=123456.6, ex2_b=123455.4;
    double dex1_a=123456.1, dex1_b=123456.0;
    double dex2_a=123456.6, dex2_b=123455.4;

    // Beyond limit of 7 digits that can be trusted
    float ex3_a=1234567.1, ex3_b=1234567.0;
    float ex4_a=1234567.6, ex4_b=1234566.4;
    double dex3_a=1234567.1, dex3_b=1234567.0;
    double dex4_a=1234567.6, dex4_b=1234566.4;

    printf("\n***** WITHIN FLOAT LIMITS SUBTRACTIVE CANCELLATION *****\n");
    printf("FLOAT  Ex 0: a-b=%f, should be 0.1, for %f-%f, but %f, %f\n", (ex0_a-ex0_b), 123.1, 123.0, ex0_a, ex0_b);

    printf("\n***** FLOAT LIMITS *****\n");
    printf("FLOAT  Ex 1: a-b=%f, should be 0.1, for %f-%f, but %f, %f\n", (ex1_a-ex1_b), 123456.1, 123456.0, ex1_a, ex1_b);
    printf("FLOAT  Ex 2: a-b=%f, should be 1.2, for %f-%f, but %f, %f\n", (ex2_a-ex2_b), 123456.6, 123455.4, ex2_a, ex2_b);
    printf("FLOAT  Ex 3: a-b=%f, should be 0.1, for %f-%f, but %f, %f\n", (ex3_a-ex3_b), 1234567.1, 1234567.0, ex3_a, ex3_b);
    printf("FLOAT  Ex 4: a-b=%f, should be 1.2, for %f-%f, but %f, %f\n", (ex4_a-ex4_b), 1234567.6, 1234566.4, ex4_a, ex4_b);
    printf("\n");
    printf("DOUBLE Ex 1: a-b=%lf, should be 0.1, for %lf-%lf=%lf\n", (dex1_a-dex1_b), dex1_a, dex1_b, 0.1);
    printf("DOUBLE Ex 2: a-b=%lf, should be 1.2, for %lf-%lf=%lf\n", (dex2_a-dex2_b), dex2_a, dex2_b, 1.2);
    printf("DOUBLE Ex 3: a-b=%lf, should be 0.1, for %lf-%lf=%lf\n", (dex3_a-dex3_b), dex3_a, dex3_b, 0.1);
    printf("DOUBLE Ex 4: a-b=%lf, should be 1.2, for %lf-%lf=%lf\n", (dex4_a-dex4_b), dex4_a, dex4_b, 1.2);


    // Within limit of 15 digits that can be trusted
    double dex0_a=1234567890123.1, dex0_b=1234567890123.0;

    // At limit of 15 digits that can be trusted
    double dex5_a=12345678901234.1, dex5_b=12345678901234.0;
    double dex6_a=12345678901234.6, dex6_b=12345678901233.4;
    long double ldex5_a=12345678901234.1L, ldex5_b=12345678901234.0L;
    long double ldex6_a=12345678901234.6L, ldex6_b=12345678901233.4L;

    // Beyond limit of 15 digits that can be trusted
    double dex7_a=123456789012345.1, dex7_b=123456789012345.0;
    double dex8_a=123456789012345.6, dex8_b=123456789012344.4;
    long double ldex7_a=123456789012345.1L, ldex7_b=123456789012345.0L;
    long double ldex8_a=123456789012345.6L, ldex8_b=123456789012344.4L;


    printf("\n***** WITHIN DOUBLE LIMITS SUBTRACTIVE CANCELLATION *****\n");
    printf("DOUBLE  Ex 0: a-b=%f, should be 0.1, for %f-%f, but %f, %f\n", (dex0_a-dex0_b), 1234567890123.1, 1234567890123.0, dex0_a, dex0_b);

    printf("\n***** DOUBLE LIMITS *****\n");
    printf("DOUBLE Ex 5: a-b=%lf, should be 0.1, for %lf-%lf, but %lf, %lf\n", (dex5_a-dex5_b), 12345678901234.1, 12345678901234.0, dex5_a, dex5_b);
    printf("DOUBLE Ex 6: a-b=%lf, should be 1.2, for %lf-%lf, but %lf, %lf\n", (dex6_a-dex6_b), 12345678901234.6, 12345668901233.4, dex6_a, dex6_b);
    printf("DOUBLE Ex 7: a-b=%lf, should be 0.1, for %lf-%lf, but %lf, %lf\n", (dex7_a-dex7_b), 123456789012345.1, 123456789012345.0, dex7_a, dex7_b);
    printf("DOUBLE Ex 8: a-b=%lf, should be 1.2, for %lf-%lf, but %lf, %lf\n", (dex8_a-dex8_b), 123456789012345.6, 123456789012344.4, dex8_a, dex8_b);
    printf("\n");

    printf("LONG DOUBLE Ex 5: a-b=%Lf, should be 0.1, for %Lf-%Lf=%lf\n", (ldex5_a-ldex5_b), ldex5_a, ldex5_b, 0.1);
    printf("LONG DOUBLE Ex 6: a-b=%Lf, should be 1.2, for %Lf-%Lf=%lf\n", (ldex6_a-ldex6_b), ldex6_a, ldex6_b, 1.2);
    printf("LONG DOUBLE Ex 7: a-b=%Lf, should be 0.1, for %Lf-%Lf=%lf\n", (ldex7_a-ldex7_b), ldex7_a, ldex7_b, 0.1);
    printf("LONG DOUBLE Ex 8: a-b=%Lf, should be 1.2, for %Lf-%Lf=%lf\n", (ldex8_a-ldex8_b), ldex8_a, ldex8_b, 1.2);

}
