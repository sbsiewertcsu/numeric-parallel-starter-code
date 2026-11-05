#include <stdio.h>

void print_binary(unsigned int number)
{
    if (number) {
        print_binary(number >> 1);
        putc((number & 1) ? '1' : '0', stdout);
    }
}

int main(void)
{
    unsigned int x=0x000000A5;
    unsigned int y=0b00000000000000000000000010100101;

    printf("x=0x%x, y=0x%x\n", x, y);

    print_binary((unsigned int)x);
    printf("\n");
    print_binary((unsigned int)y);
    printf("\n");

}
