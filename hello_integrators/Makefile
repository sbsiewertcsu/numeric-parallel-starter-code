INCLUDE_DIRS = 
LIB_DIRS = 
#CC = icc
CC = gcc
CPP = g++

CDEFS=
CFLAGS= -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)
CFLAGS2= -O0 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= hello_riemann hello_trapezoidal hello_simpson

HFILES= 
CFILES= 
HPPFILES= 
CPPFILES= hello_riemann.cpp hello_trapezoidal.cpp hello_simpson.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

hello_riemann:	hello_riemann.cpp
	$(CPP) $(CFLAGS2) -o $@ hello_riemann.cpp -lm

hello_trapezoidal:	hello_trapezoidal.cpp
	$(CPP) $(CFLAGS2) -o $@ hello_trapezoidal.cpp -lm

hello_simpson:	hello_simpson.cpp
	$(CPP) $(CFLAGS2) -o $@ hello_simpson.cpp -lm
