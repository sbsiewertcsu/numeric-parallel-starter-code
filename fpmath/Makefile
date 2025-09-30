INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc
CPP = g++

CDEFS=
CFLAGS= -O3 -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)
CFLAGS2= -O0 -g -Wall -fopenmp -std=gnu11 $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= simplemath subtractive_cancel

HFILES= 
CFILES= simplemath.c subtractive_cancel.c
HPPFILES= 
CPPFILES=

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

simplemath:	simplemath.c
	$(CC) $(CFLAGS2) -o $@ simplemath.c -lm

# gcc -std=gnu11 subtractive_cancel.c -o subcan -lquadmath -lm
subtractive_cancel:	subtractive_cancel.c
	$(CC) $(CFLAGS2) -o $@ subtractive_cancel.c -lquadmath -lm
