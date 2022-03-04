INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc
#CC=icc

CDEFS=
OMP_CFLAGS= -O0 -fopenmp $(INCLUDE_DIRS) $(CDEFS)
#OMP_CFLAGS= -O0 -qopenmp $(INCLUDE_DIRS) $(CDEFS)
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lm

HFILES= 
CFILES= timeprofiles.c timeprofiles_omp.c csvtostatic.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	timeprofiles timeprofiles_omp csvtostatic timeinterp

clean:
	-rm -f *.o *.d
	-rm -f timeprofiles timeprofiles_omp csvtostatic timeinterp

distclean:
	-rm -f *.o *.d
	-rm -f timeprofiles timeprofiles_omp csvtostatic

csvtostatic: csvtostatic.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

timeprofiles_omp: timeprofiles_omp.c
	$(CC) $(LDFLAGS) $(OMP_CFLAGS) -o $@ $@.c $(LIBS)

timeprofiles: timeprofiles.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o $(LIBS)

timeinterp: timeinterp.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
