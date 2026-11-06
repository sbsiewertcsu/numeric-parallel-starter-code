INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc
CPP = g++

CDEFS=
CFLAGS= -O0 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
CFLAGS2= -O3 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
CFLAGS3= -O0 -g -Wall -fopenmp -std=gnu11 $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= twista twistafast twist twistaprint twistaprintfast

HFILES= 
CFILES= twista.cpp twist.c
HPPFILES= 
CPPFILES=

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

twista:	twista.cpp
	$(CPP) $(CFLAGS) -o $@ twista.cpp

twistafast:	twista.cpp
	$(CPP) -DWRAPPER $(CFLAGS) -o $@ twista.cpp

twistaprint:	twista.cpp
	$(CPP) -DOUTPUT $(CFLAGS) -o $@ twista.cpp

twistaprintfast:	twista.cpp
	$(CPP) -DWRAPPER -DOUTPUT $(CFLAGS) -o $@ twista.cpp

twist:	twist.c
	$(CC) $(CFLAGS) -o $@ twist.c

