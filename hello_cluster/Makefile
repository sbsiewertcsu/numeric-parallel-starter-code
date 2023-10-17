INCLUDE_DIRS = -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/
LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
#LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/release -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
CC = mpicc
CXX = mpicxx
GCC = gcc

CDEFS=
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= greetings greetingscpp ranksum ranksumfan ranksumtree ranksumbutterfly ranksumall ranksumreduce ranksumallreduce rankscattergather piseriesreduce piseriessimp compare chatgpttrain chatgptseqtrain mpi_array

HFILES= 
CFILES= greetings.c ranksum.c ranksumfan.c ranksumtree.c ranksumbutterfly.c ranksumall.c ranksumreduce.c ranksumallreduce.c rankscattergather.c piseriesreduce.c piseriessimp.c compare.c chatgpttrain.c chatgptseqtrain.c mpi_array.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

piseriesreduce:	piseriesreduce.c
	$(CC) $(CFLAGS) -o $@ piseriesreduce.c $(LIB_DIRS)

piseriessimp: piseriessimp.c
	$(CC) $(CFLAGS) -o $@ piseriessimp.c $(LIB_DIRS) -lm

rankscattergather:	rankscattergather.c
	$(CC) $(CFLAGS) -o $@ rankscattergather.c $(LIB_DIRS)

ranksumallreduce:	ranksumallreduce.c
	$(CC) $(CFLAGS) -o $@ ranksumallreduce.c $(LIB_DIRS)

ranksumreduce:	ranksumreduce.c
	$(CC) $(CFLAGS) -o $@ ranksumreduce.c $(LIB_DIRS)

ranksumall:	ranksumall.c
	$(CC) $(CFLAGS) -o $@ ranksumall.c $(LIB_DIRS)

ranksumfan:	ranksumfan.c
	$(CC) $(CFLAGS) -o $@ ranksumfan.c $(LIB_DIRS)

ranksumtree:	ranksumtree.c
	$(CC) $(CFLAGS) -o $@ ranksumtree.c $(LIB_DIRS)

ranksumbutterfly:	ranksumbutterfly.c
	$(CC) $(CFLAGS) -o $@ ranksumbutterfly.c $(LIB_DIRS)

ranksum:	ranksum.c
	$(CC) $(CFLAGS) -o $@ ranksum.c $(LIB_DIRS)

greetingscpp:	greetings.cpp
	$(CXX) $(CFLAGS) -o $@ greetings.cpp $(LIB_DIRS)

greetings:	greetings.c
	$(CC) $(CFLAGS) -o $@ greetings.c $(LIB_DIRS)

mpi_array:	mpi_array.c
	$(CC) $(CFLAGS) -o $@ mpi_array.c $(LIB_DIRS)

compare:	compare.c trap.h sine.h
	$(CC) $(CFLAGS) -o $@ compare.c $(LIB_DIRS)

chatgpttrain:	chatgpttrain.c ex4.h
	$(CC) $(CFLAGS) -o $@ chatgpttrain.c $(LIB_DIRS)

chatgptseqtrain:	chatgptseqtrain.c ex4.h
	$(GCC) $(CFLAGS) -o $@ chatgptseqtrain.c $(LIB_DIRS)
