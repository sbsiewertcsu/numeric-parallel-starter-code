INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc
CPP=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
CPPDEFS=
CPPFLAGS= -O0 -g $(INCLUDE_DIRS) $(CPPDEFS)
LIBS= 

HFILES= 
CFILES= eratos.c
CPPFILES= eratositerator.cpp eratosrecursive.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	eratos eratositerator eratosrecursive

clean:
	-rm -f *.o *.d
	-rm -f eratos eratositerator eratosrecursive

eratos: eratos.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

eratositerator: eratositerator.o
	$(CPP) $(LDFLAGS) $(CPPFLAGS) -o $@ $@.o

eratosrecursive: eratosrecursive.o
	$(CPP) $(LDFLAGS) $(CPPFLAGS) -o $@ $@.o


depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $<

