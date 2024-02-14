INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc
CPP = g++

CDEFS=

#CFLAGS= -O3 -msse3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -msse3 -malign-double $(INCLUDE_DIRS) $(CDEFS)

CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lpthread

PRODUCT=eratos pinum largestprime eratosimp

HFILES= 
CFILES= eratos.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

eratosimp:	eratosimp.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ eratosimp.o $(LIBS)

pinum:	pinum.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ pinum.o $(LIBS)

eratos:	eratos.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ eratos.o $(LIBS)

largestprime:	largestprime.o
	$(CPP) $(LDFLAGS) $(CFLAGS) -o $@ largestprime.o $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
