#include <stdio.h>
#include <stdlib.h>

// Needed for binary I/O
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define MAX_ROWS (4096)
#define MAX_COLS (4096)

#define IMG_ROWS (920)
#define IMG_COLS (690)


// We can parse the header for more flexiblity and portabiltiy.
#define IMG_HEADER (38) // Typical Irfanview header for playing card

// Simple 1D array that I have to index to get to X,Y location
unsigned char oned_bigbuffer[MAX_ROWS*MAX_COLS];

void main(int argc, char *argv[])
{
    int fdin, fdout;
    int idx;
    int bytesread, byteswritten;

    if(argc < 3) {
        printf("Use: readnwrite <input-file> <output-file>\n");
        exit(-1);
    }

    // check for errors on open
    fdin=open(argv[1], O_RDONLY);

    // output file
    fdout=open(argv[2], O_WRONLY | O_CREAT, 0644);

    // read PGM data into a buffer
    bytesread = read(fdin, &oned_bigbuffer[0], (IMG_ROWS*IMG_COLS)+IMG_HEADER);
    printf("bytesread = %d, bytes requested=%d\n", bytesread, (IMG_ROWS*IMG_COLS)+IMG_HEADER);
    close(fdin);


    // Write back out the image
    byteswritten = write(fdout, &oned_bigbuffer[0], (IMG_ROWS*IMG_COLS)+IMG_HEADER);
    printf("byteswritten = %d, bytes requested=%d\n", byteswritten, (IMG_ROWS*IMG_COLS)+IMG_HEADER);
    close(fdout);
}
