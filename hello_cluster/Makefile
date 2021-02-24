INCLUDE_DIRS = 
LIB_DIRS = 
CC = mpicc

CDEFS=
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= greetings ranksum ranksumfan ranksumtree ranksumbutterfly ranksumall ranksumreduce ranksumallreduce rankscattergather piseriesreduce

HFILES= 
CFILES= greetings.c ranksum.c ranksumfan.c ranksumtree.c ranksumbutterfly.c ranksumall.c ranksumreduce.c ranksumallreduce.c rankscattergather.c piseriesreduce.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

piseriesreduce:	piseriesreduce.c
	$(CC) $(CFLAGS) -o $@ piseriesreduce.c

rankscattergather:	rankscattergather.c
	$(CC) $(CFLAGS) -o $@ rankscattergather.c

ranksumallreduce:	ranksumallreduce.c
	$(CC) $(CFLAGS) -o $@ ranksumallreduce.c

ranksumreduce:	ranksumreduce.c
	$(CC) $(CFLAGS) -o $@ ranksumreduce.c

ranksumall:	ranksumall.c
	$(CC) $(CFLAGS) -o $@ ranksumall.c

ranksumfan:	ranksumfan.c
	$(CC) $(CFLAGS) -o $@ ranksumfan.c

ranksumtree:	ranksumtree.c
	$(CC) $(CFLAGS) -o $@ ranksumtree.c

ranksumbutterfly:	ranksumbutterfly.c
	$(CC) $(CFLAGS) -o $@ ranksumbutterfly.c

ranksum:	ranksum.c
	$(CC) $(CFLAGS) -o $@ ranksum.c

greetings:	greetings.c
	$(CC) $(CFLAGS) -o $@ greetings.c
