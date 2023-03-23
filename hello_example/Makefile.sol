INCLUDE_DIRS =
LIB_DIRS =

MPI_INCLUDE_DIRS = -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/
MPI_LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib

CC=gcc
MPICC=mpicc

CFLAGS= -O0 -g -Wall $(INCLUDE_DIRS) $(CDEFS)
CFLAGS2= -O0 -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)

CFILES= sequential.c pthread.c ompthread.c mpirank.c riemann.c omprieman.c sol1.c sol2.c sol3.c sol4.c sol5.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	sequential pthread ompthread mpirank riemann ompriemann sol1 sol2 sol3 sol4 sol5

clean:
	-rm -f *.o *.d
	-rm -f sequential pthread ompthread mpirank riemann ompriemann sol1 sol2 sol3 sol4 sol5

sequential: sequential.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

pthread: pthread.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

sol2: sol2.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o -lpthread

sol3.o: sol3.c
	$(CC) $(CFLAGS2) -c $<

sol3: sol3.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o -lm

sol4.o: sol4.c
	$(CC) $(CFLAGS2) -c $<

sol4: sol4.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o -lm

ompthread.o: ompthread.c
	$(CC) $(CFLAGS2) -c $<

ompthread: ompthread.o
	$(CC) $(LDFLAGS) $(CFLAGS2) -o $@ $@.o

sol5:  sol5.c
	$(MPICC) $(CFLAGS) -o $@ sol5.c $(MPI_LIB_DIRS) -lm

sol1:  sol1.c
	$(MPICC) $(CFLAGS) -o $@ sol1.c $(MPI_LIB_DIRS) -lm

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
