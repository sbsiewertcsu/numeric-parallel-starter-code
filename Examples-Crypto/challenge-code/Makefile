INCLUDE_DIRS = 
LIB_DIRS = 

CDEFS=
CFLAGS= -O -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

PRODUCT2=prgcoding

HFILES=
CFILES= prgcoding.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all: prgcoding

clean:
	-rm -f *.o *.d *.exe prgcoding

prgcoding: prgcoding.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

depend:

.c.o:
	$(CC) -MD $(CFLAGS) -c $<
