INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=
CFLAGS= -O0 -fopenmp $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O0 -g -pg $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= gewpp.c gewppoff.c gaussjord.c lufact.c gsit.c gsitbad.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	gewpp gewppoff gaussjord gaussjordomp lufact gsit gsit2 gsit5 gsitbad gewppomp vmult testgen transpose

clean:
	-rm -f *.o *.d
	-rm -f gewpp gewppoff gaussjord gaussjordomp lufact gsit gewppomp vmult testgen gsit2 gsit5 gsitbad transpose

distclean:
	-rm -f *.o *.d
	-rm -f gewpp gewppoff gaussjord lufact gsit

gewppomp: gewppomp.o
	$(CC) $(LDFLAGS) -fopenmp $(CFLAGS) -o $@ $@.o

gewpp: gewpp.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gewppoff: gewppoff.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gaussjordomp: gaussjordomp.o
	$(CC) $(LDFLAGS) -fopenmp $(CFLAGS) -o $@ $@.o

gaussjord: gaussjord.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

lufact: lufact.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gsit5: gsit5.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gsit2: gsit2.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gsit: gsit.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

gsitbad: gsitbad.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

transpose: transpose.o
	$(CC) $(LDFLAGS) -fopenmp $(CFLAGS) -o $@ $@.o

vmult: vmult.o
	$(CC) $(LDFLAGS) -fopenmp $(CFLAGS) -o $@ $@.o

testgen: testgen.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
