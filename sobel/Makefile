INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc

CDEFS=
CFLAGS= -O3 $(INCLUDE_DIRS) $(CDEFS)
LIBS=-lm

PRODUCT=sobel

HFILES= 
CFILES= sobel.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~ *SobelTransform*.bmp
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

sobel:	sobel.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ sobel.o $(LIBS)

.c.o:
	$(CC) $(CFLAGS) -c $<
