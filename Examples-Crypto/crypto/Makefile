INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc
CPP = g++

CDEFS= 
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT=crypt transcrypt subcrypt

HFILES=
CFILES=

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.d
	-rm -f ${PRODUCT} ${GARBAGE}

crypt:	crypt.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ crypt.o

transcrypt:	transcrypt.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ transcrypt.o

subcrypt:	subcrypt.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ subcrypt.o


depend:

.cpp.o:
	$(CPP) -MD $(CFLAGS) -c $<

.c.o:
	$(CC) -MD $(CFLAGS) -c $<
