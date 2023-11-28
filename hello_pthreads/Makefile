INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc

CDEFS=
CFLAGS= -g -O0 -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lpthread

PRODUCT= hello_pthread pthreadpi fibsum fib

HFILES= 
CFILES= hello_omp.c omppi.c fibsum.c fib.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

fib:	fib.c
	$(CC) $(CFLAGS) -o $@ fib.c $(LIBS)

fibsum:	fibsum.c
	$(CC) $(CFLAGS) -o $@ fibsum.c $(LIBS)

pthreadpi:	pthreadpi.c
	$(CC) $(CFLAGS) -o $@ pthreadpi.c $(LIBS)

hello_pthread:	hello_pthread.c
	$(CC) $(CFLAGS) -o $@ hello_pthread.c $(LIBS)
