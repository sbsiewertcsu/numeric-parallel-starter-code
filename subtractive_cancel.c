#include <stdio.h>

// https://en.wikipedia.org/wiki/IEEE_754
// https://en.wikipedia.org/wiki/Extended_precision#x86_extended_precision_format

#define SINGLE_ITERATIONS 25 // IEEE 754 significand is 23 bits
#define DOUBLE_ITERATIONS 55 // IEEE 754 significand is 52 bits
#define LONG_DOUBLE_ITERATIONS 65 // IEEE 754 significand is 63 bits

float single_test=0.0, single_addend=1.0;
double double_test=0.0, double_addend=1.0;
long double long_double_test=0.0, long_double_addend=1.0;

int main(void)
{
    int idx;

    for(idx=0; idx < SINGLE_ITERATIONS; idx++)
    {
        single_addend = single_addend / 2.0;
        single_test = 1.0 + single_addend;


        // This test will succeed when 2^(-k) can't be represented
        if(single_test == 1.0)
        {
                printf("SINGLE PRECISION digit limit found: ");
                printf("LIMIT for 1+2^(-k)=1.0 @ k=%2.2d\n", idx+1);
        }
        else
        {
                printf("@ k=%2.2d, 1+2^(-k)=%.8f\n", idx+1, single_test);
        }

    }


    for(idx=0; idx < DOUBLE_ITERATIONS; idx++)
    {
            double_addend = double_addend / 2.0;
            double_test = 1.0 + double_addend;

            // This test will succeed when 2^(-k) can't be represented
            if(double_test == 1.0)
            {
                    printf("DOUBLE PRECISON digit limit found: ");
                    printf("LIMIT for 1+2^(-k)=1.0 @ k=%2.2d\n", idx+1);
            }
            else
            {
                    printf("@ k=%2.2d, 1+2^(-k)=%.16f\n", idx+1, double_test);
            }

    }


    for(idx=0; idx < LONG_DOUBLE_ITERATIONS; idx++)
    {
            long_double_addend = long_double_addend / 2.0;
            long_double_test = 1.0 + long_double_addend;

            // This test will succeed when 2^(-k) can't be represented
            if(long_double_test == 1.0)
            {
                    printf("DOUBLE PRECISON digit limit found: ");
                    printf("LIMIT for 1+2^(-k)=1.0 @ k=%2.2d\n", idx+1);
            }
            else

            {
                    printf("@ k=%2.2d, 1+2^(-k)=%.34Lf\n", idx+1, long_double_test);
            }

    }


    // Testing for proper support of special values NaN and Inf in IEEE Floating Point
    double zero=0.0, one=1.0;

    printf("Nan and Inf tests:\n");
    printf("zero/zero = %lf\n", zero/zero);
    printf("one/zero = %lf\n", one/zero);
    printf("one/zero = %lf\n", -(zero/zero));
    printf("one/zero = %lf\n", -(one/zero));

}
