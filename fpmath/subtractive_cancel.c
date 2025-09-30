// Compile this with:
//
// gcc -std=gnu11 subtractive_cancel.c -o subcan -lquadmath
//
// gcc subtractive_cancel.c -o subcan -lm
//
// OR REMOVED QUAD DOUBLE on an older compiler
//
#include <stdio.h>
#include <math.h>
#include <quadmath.h>  //include the for quad double testing

// https://en.wikipedia.org/wiki/IEEE_754
// https://en.wikipedia.org/wiki/Extended_precision#x86_extended_precision_format

#define SINGLE_ITERATIONS 25 // IEEE 754 significand is 23 bits
#define DOUBLE_ITERATIONS 55 // IEEE 754 significand is 52 bits
#define LONG_DOUBLE_ITERATIONS 65 // IEEE 754 significand is 63 bits

//https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format
#define QUAD_DOUBLE_ITERATIONS 113 // IEEE 754 significand is 113 bits
#define QUAD_TEST

float single_test=0.0, single_addend=1.0;
double double_test=0.0, double_addend=1.0;
long double long_double_test=0.0, long_double_addend=1.0;

int main(void)
{
    int idx;

    // FLOAT TEST for limit of Precision and mantissa size in bits
    for(idx=0; idx < SINGLE_ITERATIONS; idx++)
    {
        single_addend = single_addend / 2.0;
        single_test = 1.0 + single_addend;


        // This test will succeed when 2^(-k) can't be represented
        if(single_test == 1.0)
        {
            printf("SINGLE PRECISION digit limit found: ");
            printf("LIMIT for 1+2^(-k)=1.0 @ k=%2.2d\n", idx+1);
            printf("DIGITS YOU CAN TRUST = %d\n", (int)floor((double)(idx) * log10(2.0)) + 1);
            break;
        }
        else
        {
            printf("@ k=%2.2d, 1+2^(-k)=%.8f\n", idx+1, single_test);
        }
    }
    printf("\n");


    // DOUBLE TEST for limit of Precision and mantissa size in bits
    for(idx=0; idx < DOUBLE_ITERATIONS; idx++)
    {
        double_addend = double_addend / 2.0;
        double_test = 1.0 + double_addend;

        // This test will succeed when 2^(-k) can't be represented
        if(double_test == 1.0D)
        {
            printf("DOUBLE PRECISON digit limit found: ");
            printf("LIMIT for 1+2^(-k)=1.0 @ k=%2.2d\n", idx+1);
            printf("DIGITS YOU CAN TRUST = %d\n", (int)floor((double)(idx) * log10(2.0)) + 1);
            break;
        }
        else
        {
             printf("@ k=%2.2d, 1+2^(-k)=%.16f\n", idx+1, double_test);
        }
    }
    printf("\n");


    // LONG DOUBLE TEST for limit of Precision and mantissa size in bits
    for(idx=0; idx < LONG_DOUBLE_ITERATIONS; idx++)
    {
        long_double_addend = long_double_addend / 2.0;
        long_double_test = 1.0 + long_double_addend;

        // This test will succeed when 2^(-k) can't be represented
        if(long_double_test == 1.0L)
        {
            printf("LONG DOUBLE PRECISON digit limit found: ");
            printf("LIMIT for 1+2^(-k)=1.0 @ k=%2.2d\n", idx+1);
            printf("DIGITS YOU CAN TRUST = %d\n", (int)floor((double)(idx) * log10(2.0)) + 1);
            break;
        }
        else
        {
            printf("@ k=%2.2d, 1+2^(-k)=%.34Lf\n", idx+1, long_double_test);
        }
    }
    printf("\n");

#ifdef QUAD_TEST
    // ADD QUAD DOUBLE TEST here
    //
    // Gemini: GCC provides the non-standard type __float128 for 128-bit quadruple-precision floating-point values
    //
    __float128 qdouble_addend=1.0Q, qdouble_test=0.0Q;
    char buffer[128];

    for(idx=0; idx < QUAD_DOUBLE_ITERATIONS; idx++)
    {
        qdouble_addend = qdouble_addend / 2.0Q;
        qdouble_test = 1.0Q + qdouble_addend;

        // This test will succeed when 2^(-k) can't be represented
        if(qdouble_test == 1.0Q)
        {
            printf("QUAD DOUBLE PRECISON digit limit found: ");
            printf("LIMIT for 1+2^(-k)=1.0 @ k=%2.2d\n", idx+1);
            printf("DIGITS YOU CAN TRUST = %d\n", (int)floor((double)(idx) * log10(2.0)) + 1);
            break;
        }
        else
        {
            quadmath_snprintf(buffer, sizeof(buffer), "%.34Qf", qdouble_test);
            printf("@ k=%2.2d, 1+2^(-k)=%s\n", idx+1, buffer);
        }
    }
    printf("\n");
#endif

    // DOUBLE special case testing
    //
    // Testing for proper support of special values NaN and Inf in IEEE Floating Point
    double zero=0.0, one=1.0;

    printf("Nan and Inf tests:\n");
    printf("zero/zero = %lf\n", zero/zero);
    printf("one/zero = %lf\n", one/zero);
    printf("-(zero/zero) = %lf\n", -(zero/zero));
    printf("-(one/zero) = %lf\n", -(one/zero));

}
