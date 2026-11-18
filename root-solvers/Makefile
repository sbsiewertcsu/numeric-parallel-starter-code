INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc
CPP = g++

CDEFS=
CFLAGS= -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= bisection regulafalsi newton bruteroot brents rf_compare

HFILES= 
CFILES= regulafalsi.c newton.c bruteroot.c brents.c rf_compare.c
CPPFILES= bisection.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

bruteroot:	bruteroot.c
	$(CC) $(CFLAGS) -o $@ bruteroot.c -lm

bisection:	bisection.cpp
	$(CPP) $(CFLAGS) -o $@ bisection.cpp -lm

regulafalsi:	regulafalsi.c
	$(CC) $(CFLAGS) -o $@ regulafalsi.c -lm

rf_compare:	rf_compare.c
	$(CC) $(CFLAGS) -o $@ rf_compare.c -lm

newton:	newton.c
	$(CC) $(CFLAGS) -o $@ newton.c -lm

brents:	brents.c
	$(CC) $(CFLAGS) -o $@ brents.c -lm


