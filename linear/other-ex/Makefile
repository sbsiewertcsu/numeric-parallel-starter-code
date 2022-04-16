INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc
CPP=g++

CDEFS=
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O0 -g -pg $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= gauss_seidel_iterative.c gauss_elim2.c
CPPFILES= gausselim.cpp gausselim_geeks.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:    gsit gewpp gewpp2 gewpp3

clean:
	-rm -f *.o *.d
	-rm -f gsit gewpp gewpp2 gewpp3

distclean:
	-rm -f *.o *.d
	-rm -f gsit

gsit: gauss_seidel_iterative.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ gauss_seidel_iterative.o

gewpp2: gausselim2.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ gausselim2.o

gewpp: gausselim.o
	$(CPP) $(LDFLAGS) $(CFLAGS) -o $@ gausselim.o

gewpp3: gausselim_geeks.o
	$(CPP) $(LDFLAGS) $(CFLAGS) -o $@ gausselim_geeks.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
