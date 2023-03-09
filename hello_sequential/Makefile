INCLUDE_DIRS =
LIB_DIRS =

MPI_INCLUDE_DIRS = -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/
MPI_LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib

CC=gcc
MPICC=mpicc

CFLAGS= -O0 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
CFLAGS2= -O0 -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)

CFILES= sequential.c pthread.c ompthread.c mpirank.c riemann.c omprieman.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	sequential pthread ompthread mpirank riemann ompriemann

clean:
	-rm -f *.o *.d
	-rm -f sequential pthread ompthread mpirank riemann ompriemann

sequential: sequential.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

pthread: pthread.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

ompthread.o: ompthread.c
	$(CC) $(CFLAGS2) -c $<

ompthread: ompthread.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o

mpirank:  mpirank.c
	$(MPICC) $(CFLAGS) -o $@ mpirank.c $(MPI_LIB_DIRS)

riemann.o: riemann.c
	$(CC) $(CFLAGS) -c $<

riemann: riemann.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lm

ompriemann.o: ompriemann.c
	$(CC) $(CFLAGS2) -c $<

ompriemann: ompriemann.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o -lm

.c.o:
	$(CC) $(CFLAGS) -c $<
