INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc
CPP=g++
#CC=icc

CDEFS=
CFLAGS= -std=c++11 $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= pthread.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	bigmath

clean:
	-rm -f *.o *.d
	-rm -f bigmath

bigmath: bigmath.o
	$(CPP) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.cpp.o:
	$(CPP) $(CFLAGS) -c $<
