INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=
#CFLAGS= -O0 -g -fopenmp $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O2 -g -fopenmp $(INCLUDE_DIRS) $(CDEFS)
CFLAGS= -O3 -g -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= eratos.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	assocmem

clean:
	-rm -f *.o *.d
	-rm -f assocmem

assocmem: assocmem.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $<

