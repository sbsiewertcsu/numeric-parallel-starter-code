INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc

CDEFS= -std=c99
#CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O0 -msse3 -malign-double $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O2 -msse3 -malign-double $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -msse3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -mssse3 $(INCLUDE_DIRS) $(CDEFS)
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#LIBS=-lpthread
LIBS= -lm

PRODUCT=fft

HFILES= 
CFILES= fft.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

fft:	fft.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ fft.o $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
