INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc

CDEFS=
CFLAGS= -O0 -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)
CFLAGS2= -O0 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= hello_omp hello_pthread omppi fibsum fib piseries piseriesompfunct incdec fibgpt

HFILES= 
CFILES= hello_omp.c omppi.c fibsum.c fib.c piseries.c piseriesompfunct.c incdec.c fibgpt.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

incdec:	incdec.c
	$(CC) $(CFLAGS) -fopenmp -o $@ incdec.c

fib:	fib.c
	$(CC) $(CFLAGS) -o $@ fib.c

fibgpt:	fibgpt.c
	$(CC) $(CFLAGS) -o $@ fibgpt.c

piseriesompfunct:	piseriesompfunct.c
	$(CC) $(CFLAGS) -o $@ piseriesompfunct.c -lm

piseries:	piseries.c
	$(CC) $(CFLAGS) -o $@ piseries.c

fibsum:	fibsum.c
	$(CC) $(CFLAGS) -o $@ fibsum.c

omppi:	omppi.c
	$(CC) $(CFLAGS) -o $@ omppi.c

hello_omp:	hello_omp.c
	$(CC) $(CFLAGS) -o $@ hello_omp.c

hello_pthread:	hello_pthread.c
	$(CC) $(CFLAGS2) -o $@ hello_pthread.c -lpthread -lrt
