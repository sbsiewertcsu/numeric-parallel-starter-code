INCLUDE_DIRS = 
LIB_DIRS = 
CC = g++

CDEFS=
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -msse3 -malign-double $(INCLUDE_DIRS) $(CDEFS)


all:	eratos eratositerator eratosrecursive
#all:	eratos

clean:
	-rm -f *.o *.NEW *~
	-rm -f eratos eratositerator eratosrecursive ${DERIVED} ${GARBAGE}

eratos:	eratos.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ eratos.o $(LIBS)

eratositerator:	eratositerator.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ eratositerator.o $(LIBS)

eratosrecursive:	eratosrecursive.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ eratosrecursive.o $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
