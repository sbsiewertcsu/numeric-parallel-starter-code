#include <iostream>
#include <gmpxx.h>

int main()
{
    mpz_class a(
        "12345678901234567890123456789012345678901234567890");

    mpz_class b(
        "9876543210987654321098765432109876543210987654321");

    mpz_class sum       = a + b;
    mpz_class diff      = a - b;
    mpz_class product   = a * b;
    mpz_class quotient  = a / b;
    mpz_class remainder = a % b;

    std::cout << "a = " << a << '\n';
    std::cout << "b = " << b << '\n';
    std::cout << "sum = " << sum << '\n';
    std::cout << "difference = " << diff << '\n';
    std::cout << "product = " << product << '\n';
    std::cout << "quotient = " << quotient << '\n';
    std::cout << "remainder = " << remainder << '\n';

    return 0;
}
