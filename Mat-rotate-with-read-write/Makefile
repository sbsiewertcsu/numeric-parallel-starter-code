INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=
CFLAGS= -O0 -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= matrotate.c dirlist.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	matrotate dirlist

clean:
	-rm -f *.o *.d
	-rm -f matrotate dirlist

distclean:
	-rm -f *.o *.d
	-rm -f matrotate dirlist

matrotate: matrotate.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

dirlist: dirlist.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
