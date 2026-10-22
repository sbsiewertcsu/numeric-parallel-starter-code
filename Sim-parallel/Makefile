INCLUDE_DIRS = -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/
LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
CC=gcc
CPP=g++
MPICC=mpicc
#CC=icc

CDEFS=
OMP_CFLAGS= -O0 -fopenmp $(INCLUDE_DIRS) $(CDEFS)
#OMP_CFLAGS= -O0 -qopenmp $(INCLUDE_DIRS) $(CDEFS)
CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
#CFLAGS= -O0 $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lm

HFILES= 
CFILES= simtrainideal.c simtrain_omp.c simtrainideal_omp.c csvtostatic.c hello_rk87.cpp hello_rk89.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	simtrainideal simtrain_omp simtrainideal_omp csvtostatic hello_rk87 hello_rk89

clean:
	-rm -f *.o *.d
	-rm -f simtrainideal simtrain_omp simtrainideal_omp csvtostatic hello_rk87 hello_rk89

distclean:
	-rm -f *.o *.d
	-rm -f simtrainideal simtrain_omp simtrainideal_omp csvtostatic

simtrain_omp: simtrain_omp.c
	$(CC) $(LDFLAGS) $(OMP_CFLAGS) -o $@ $@.c $(LIBS)

simtrainideal_omp: simtrainideal_omp.c
	$(CC) $(LDFLAGS) $(OMP_CFLAGS) -o $@ $@.c $(LIBS)

simtrainideal: simtrainideal.c
	$(MPICC) $(LDFLAGS) $(OMP_CFLAGS) -o $@ $@.c $(LIBS)

hello_rk89: hello_rk89.cpp
	$(CPP) $(LDFLAGS) -o $@ $@.cpp $(LIBS)

hello_rk87: hello_rk87.cpp
	$(CPP) $(LDFLAGS) -o $@ $@.cpp $(LIBS)

csvtostatic: csvtostatic.c
	$(CC) $(LDFLAGS) -o $@ $@.c $(LIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CPP) $(CFLAGS) -c $<
