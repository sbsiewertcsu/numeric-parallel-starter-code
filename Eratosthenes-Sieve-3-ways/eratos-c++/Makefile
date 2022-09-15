INCLUDE_DIRS = 
LIB_DIRS = 
CPP=g++

CPPDEFS=
CPPFLAGS= -O0 -g $(INCLUDE_DIRS) $(CPPDEFS)
LIBS= 

HFILES= 
CPPFILES= eratos.cpp eratositerator.cpp eratosrecursive.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all:	eratos eratositerator eratosrecursive
#all:	eratos

clean:
	-rm -f *.o *.d
	-rm -f eratos eratositerator eratosrecursive

eratos: eratos.o
	$(CPP) $(LDFLAGS) $(CFLAGS) -o $@ $@.o

eratositerator: eratositerator.o
	$(CPP) $(LDFLAGS) $(CPPFLAGS) -o $@ $@.o

eratosrecursive: eratosrecursive.o
	$(CPP) $(LDFLAGS) $(CPPFLAGS) -o $@ $@.o


depend:

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $<

