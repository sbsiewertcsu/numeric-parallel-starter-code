INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc

CDEFS=
CFLAGS= -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= hello_omp omppi fibsum fib

HFILES= 
CFILES= hello_omp.c omppi.c fibsum.c fib.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

fib:	fib.c
	$(CC) $(CFLAGS) -o $@ fib.c

fibsum:	fibsum.c
	$(CC) $(CFLAGS) -o $@ fibsum.c

omppi:	omppi.c
	$(CC) $(CFLAGS) -o $@ omppi.c

hello_omp:	hello_omp.c
	$(CC) $(CFLAGS) -o $@ hello_omp.c
