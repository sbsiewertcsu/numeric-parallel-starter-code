INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= gewpp.c gewppoff.c gaussjord.c lufact.c gsit.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	gewpp gewppoff gaussjord lufact gsit

clean:
	-rm -f *.o *.d
	-rm -f gewpp gewppoff gaussjord lufact gsit

distclean:
	-rm -f *.o *.d
	-rm -f gewpp gewppoff gaussjord lufact gsit

gewpp: gewpp.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gewppoff: gewppoff.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gaussjord: gaussjord.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

lufact: lufact.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gsit: gsit.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
