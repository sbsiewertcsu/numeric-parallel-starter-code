INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc
CPP = g++

CDEFS=

#CFLAGS= -O3 -msse3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -msse3 -malign-double $(INCLUDE_DIRS) $(CDEFS)

CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lpthread

PRODUCT=erast pinum largestprime erastsimp

HFILES= 
CFILES= erast.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

erastsimp:	erastsimp.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ erastsimp.o $(LIBS)

pinum:	pinum.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ pinum.o $(LIBS)

erast:	erast.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ erast.o $(LIBS)

largestprime:	largestprime.o
	$(CPP) $(LDFLAGS) $(CFLAGS) -o $@ largestprime.o $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
