INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc

CDEFS=
CFLAGS= -O3 $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= pthread.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	pthread pthread_sem

clean:
	-rm -f *.o *.d
	-rm -f pthread pthread_sem

distclean:
	-rm -f *.o *.d
	-rm -f pthread

pthread_sem: pthread_sem.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

pthread: pthread.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
