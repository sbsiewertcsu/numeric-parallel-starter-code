INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= matrotate.c readnwrite.c readntrans.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	matrotate readnwrite readntrans

clean:
	-rm -f *.o *.d
	-rm -f matrotate readnwrite readntrans

distclean:
	-rm -f *.o *.d
	-rm -f matrotate

readnwrite: readnwrite.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

readntrans: readntrans.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

matrotate: matrotate.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
