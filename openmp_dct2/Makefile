INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=

#CFLAGS = -DDEBUG
CFLAGS = -O0 -fopenmp
LDFLAGS = -lm

HFILES=
CFILES= matrotate.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	dct2 ompdct2

clean:
	-rm -f *.o *.d
	-rm -f dct2 ompdct2

distclean:
	-rm -f *.o *.d
	-rm -f dct2 ompdct2

ompdct2: ompdct2.o
	$(CC) $(CFLAGS) -o $@  $@.o $(LDFLAGS)

dct2: dct2.o
	$(CC) $(CFLAGS) -o $@  $@.o $(LDFLAGS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

