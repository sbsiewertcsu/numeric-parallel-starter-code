What if I try to compile on my Linux system and get:

bigmathgmp.cpp:2:10: fatal error: gmpxx.h: No such file or directory

Yes. That error means the GMP C++ development headers are not installed or are not in the compiler's include path. gmpxx.h comes with the GMP development package, not just the runtime library.

On Ubuntu/Debian, install it with:
sudo apt update
sudo apt install libgmp-dev

Then verify:
ls /usr/include/gmpxx.h

and:
dpkg -L libgmp-dev | grep gmpxx.h
