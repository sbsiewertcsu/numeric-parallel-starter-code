INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= matrotate.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	matrotate

clean:
	-rm -f *.o *.d
	-rm -f matrotate

distclean:
	-rm -f *.o *.d
	-rm -f matrotate

matrotate: matrotate.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
