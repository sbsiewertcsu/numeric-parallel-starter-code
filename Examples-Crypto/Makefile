INCLUDE_DIRS = 
LIB_DIRS = 

CDEFS=
CFLAGS= -O -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

PRODUCT=enigma3 subcrypt lutcipher

HFILES=
CFILES= enigma3.c subcrypt.c lutcipher.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all: enigma3 subcrypt lutcipher

clean:
	-rm -f *.o *.d *.exe enigma3 subcrypt lutcipher

subcrypt: subcrypt.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ subcrypt.o $(LIBS)

enigma3: enigma3.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ enigma3.o $(LIBS)

lutcipher: lutcipher.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ lutcipher.o $(LIBS)

depend:

.c.o:
	$(CC) -MD $(CFLAGS) -c $<
