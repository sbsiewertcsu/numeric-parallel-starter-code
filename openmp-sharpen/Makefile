INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc

CDEFS=
#CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O1 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O2 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -mcpu=cortex-a7 -mfpu=neon-vfpv4 $(INCLUDE_DIRS) $(CDEFS)
CFLAGS= -O3 -fopenmp $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= sharpen

HFILES= 
CFILES= sharpen.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~ *sharp*.ppm
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

sharpen:	sharpen.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ sharpen.o $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
