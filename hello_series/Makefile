INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc

CDEFS=
CFLAGS= -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)
CFLAGS2= -O0 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= piseries expseries expseries2 cosseries sinseries

HFILES= 
CFILES= piseries.c expseries.c expseries2.c cosseries.c sinseries.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

expseries2:	expseries2.c
	$(CC) $(CFLAGS2) -o $@ expseries2.c -lm

expseries:	expseries.c
	$(CC) $(CFLAGS) -o $@ expseries.c -lm

cosseries:	cosseries.c
	$(CC) $(CFLAGS) -o $@ cosseries.c -lm

sinseries:	sinseries.c
	$(CC) $(CFLAGS) -o $@ sinseries.c -lm

piseries:	piseries.c
	$(CC) $(CFLAGS) -o $@ piseries.c
