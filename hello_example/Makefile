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
	-rm -f sequential pthread ompthread mpirank riemann ompriemann riemann.o ompriemann.o

sequential: sequential.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

pthread: pthread.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

lange2: lange2.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

lange3.o: lange3.c
	$(CC) $(CFLAGS2) -c $<

lange3: lange3.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o -lm

lange4.o: lange4.c
	$(CC) $(CFLAGS2) -c $<

lange4: lange4.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o -lm

ompthread.o: ompthread.c
	$(CC) $(CFLAGS2) -c $<

ompthread: ompthread.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o

lange5:  lange5.c
	$(MPICC) $(CFLAGS) -o $@ lange5.c $(MPI_LIB_DIRS) -lm

carson1:  carson1.c
	$(MPICC) $(CFLAGS) -o $@ carson1.c $(MPI_LIB_DIRS) -lm

mpirank:  mpirank.c
	$(MPICC) $(CFLAGS) -o $@ mpirank.c $(MPI_LIB_DIRS)

riemann.o: riemann.c
	$(CC) $(CFLAGS2) -c $<

riemann: riemann.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lm

ompriemann.o: ompriemann.c
	$(CC) $(CFLAGS2) -c $<

ompriemann: ompriemann.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o -lm

.c.o:
	$(CC) $(CFLAGS) -c $<
