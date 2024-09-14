INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=
CFLAGS= -O0 -g -fopenmp $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O2 -g -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= eratos.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	eratos eratosomp primescale

clean:
	-rm -f *.o *.d
	-rm -f eratos eratosomp primescale

primescale: primescale.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

eratosomp: eratosomp.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

eratos: eratos.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $<

