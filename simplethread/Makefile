INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc
#CC=icc

CDEFS=
CFLAGS= -O0 -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= pthread.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	pthread pthread2 pthread3 pthread4

clean:
	-rm -f *.o *.d
	-rm -f pthread pthread2 pthread3 pthread4

pthread: pthread.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

pthread2: pthread2.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

pthread3: pthread3.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

pthread4: pthread4.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
