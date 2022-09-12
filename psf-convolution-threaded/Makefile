INCLUDE_DIRS = 
LIB_DIRS = 
CC = gcc

CDEFS=
#CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O1 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O2 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O3 -mcpu=cortex-a7 -mfpu=neon-vfpv4 $(INCLUDE_DIRS) $(CDEFS)
CFLAGS= -O3 -mssse3 $(INCLUDE_DIRS) $(CDEFS)
LIBS=-lpthread

PRODUCT=sharpen_grid sharpen_slice sharpen

HFILES= 
CFILES= sharpen_grid.c sharpen_slice.c sharpen.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~ *sharp*.ppm
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

sharpen_slice:	sharpen_slice.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ sharpen_grid.o $(LIBS)

sharpen_grid:	sharpen_grid.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ sharpen_grid.o $(LIBS)

sharpen:	sharpen.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ sharpen.o $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
