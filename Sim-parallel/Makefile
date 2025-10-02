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

all:	simtrain_omp csvtostatic

clean:
	-rm -f *.o *.d
	-rm -f simtrain_omp csvtostatic

distclean:
	-rm -f *.o *.d
	-rm -f simtrain_omp csvtostatic

simtrain_omp: simtrain_omp.c
	$(CC) $(LDFLAGS) $(OMP_CFLAGS) -o $@ $@.c $(LIBS)

csvtostatic: csvtostatic.c
	$(CC) $(LDFLAGS) -o $@ $@.c $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
