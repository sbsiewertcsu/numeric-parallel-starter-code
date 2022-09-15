INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc

CDEFS=
#CFLAGS= -O3 -msse3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -msse3 -malign-double $(INCLUDE_DIRS) $(CDEFS)
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)

LIBS= -lpthread

PRODUCT=eratospthread eratos

HFILES= 
CFILES= eratos.c eratospthread.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

eratos: eratos.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ eratos.o $(LIBS)

eratospthread: eratospthread.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ eratospthread.o $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
