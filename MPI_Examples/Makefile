INCLUDE_DIRS = -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/
LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
MPICC = mpicc
CC = gcc

CDEFS=
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
MPIFLAGS= -g -Wall -O2 $(INCLUDE_DIRS) $(CDEFS)
MPIFLAGS_NOWARN= -g -O2 $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= bubble odd_even vector_add trap mpi_many_msgs mpi_hello mpi_mat_vect_mult mpi_mat_vect_time mpi_odd_even mpi_output mpi_vector_add mpi_trap1 mpi_trap2 mpi_trap3 mpi_trap4 testex4

HFILES= 
MPIFILES= mpi_many_msgs.c mpi_hello.c mpi_mat_vect_mult.c mpi_mat_vect_time.c mpi_odd_even.c mpi_output.c mpi_vector_add.c mpi_trap1.c mpi_trap2.c mpi_trap3.c mpi_trap4.
CFILES= bubble.c odd_even.c vector_add.c mpi_trap1.c mpi_trap2.c mpi_trap3.c mpi_trap4.c trap.c testex4.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

mpi_many_msgs:	mpi_many_msgs.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_many_msgs.c $(LIB_DIRS)

mpi_hello:	mpi_hello.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_hello.c $(LIB_DIRS)

mpi_mat_vect_mult:	mpi_mat_vect_mult.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_mat_vect_mult.c $(LIB_DIRS)

mpi_mat_vect_time:	mpi_mat_vect_time.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_mat_vect_time.c $(LIB_DIRS)

mpi_odd_even:	mpi_odd_even.c
	$(MPICC) $(MPIFLAGS_NOWARN) -o $@ mpi_odd_even.c $(LIB_DIRS)

mpi_output:	mpi_output.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_output.c $(LIB_DIRS)

mpi_vector_add:	mpi_vector_add.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_vector_add.c $(LIB_DIRS)

mpi_trap1:	mpi_trap1.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_trap1.c $(LIB_DIRS) -lm

mpi_trap2:	mpi_trap2.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_trap2.c $(LIB_DIRS)

mpi_trap3:	mpi_trap3.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_trap3.c $(LIB_DIRS) -lm

mpi_trap4:	mpi_trap4.c
	$(MPICC) $(MPIFLAGS) -o $@ mpi_trap4.c $(LIB_DIRS) -lm

bubble:	bubble.c
	$(CC) $(CFLAGS) -o $@ bubble.c $(LIB_DIRS)

odd_even:	odd_even.c
	$(CC) $(CFLAGS) -o $@ odd_even.c $(LIB_DIRS)

vector_add:	vector_add.c
	$(CC) $(CFLAGS) -o $@ vector_add.c $(LIB_DIRS)

testex4:	testex4.c
	$(CC) $(CFLAGS) -o $@ testex4.c $(LIB_DIRS) -lm

trap:	trap.c
	$(CC) $(CFLAGS) -o $@ trap.c $(LIB_DIRS)


