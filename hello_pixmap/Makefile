INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc

CDEFS=
CFLAGS= -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)
CFLAGS2= -O0 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= ppmmerge pgmmerge pgmmerge2 ppminout

HFILES= 
CFILES= ppmmerge.c pgmmerge.c pgmmerge2.c ppminout.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~ hello-world.ppm hello-world.pgm
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

ppmmerge:	ppmmerge.c
	$(CC) $(CFLAGS) -o $@ ppmmerge.c

pgmmerge:	pgmmerge.c
	$(CC) $(CFLAGS) -o $@ pgmmerge.c

pgmmerge2:	pgmmerge2.c
	$(CC) $(CFLAGS) -o $@ pgmmerge2.c

ppminout:	ppminout.c
	$(CC) $(CFLAGS) -o $@ ppminout.c
