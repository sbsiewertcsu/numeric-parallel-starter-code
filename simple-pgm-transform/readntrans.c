#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Needed for binary I/O
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define MAX_ROWS (4096)
#define MAX_COLS (4096)

#define IMG_ROWS (920)
#define IMG_COLS (690)

#define SATURATION (255)
#define TRANSFORM

// We can parse the header for more flexiblity and portabiltiy.
//
// See http://www.ecst.csuchico.edu/~sbsiewert/csci551/code/brighten_compare/c-brighten/
// for an example of header parsing and brighten/contrast transformation.
//
#define IMG_HEADER (38) // Typical Irfanview header for playing card
char header_str[IMG_HEADER];

// Simple 1D array that I have to index to get to X,Y location
unsigned char oned_bigbuffer[MAX_ROWS*MAX_COLS];

// Nicer 2D array that compiler indexes to get to X,Y location
unsigned char twod_bigbuffer[MAX_ROWS][MAX_COLS];

void main(int argc, char *argv[])
{
    int fdin, fdout;
    int rowidx, colidx, idx;
    int bytesread, byteswritten;

    if(argc < 3)
    {
        printf("Use: readnwrite <input-file> <output-file>\n");
        exit(-1);
    }

    // check for errors on open
    fdin=open(argv[1], O_RDONLY);

    // output file
    fdout=open(argv[2], O_WRONLY | O_CREAT, 0644);


     // read PGM (or PPM) data into a buffer, 1D for simple single read or 2D 
     // reading rows of data for a more useful data structure with the image
     bytesread = read(fdin, &oned_bigbuffer[0], (IMG_ROWS*IMG_COLS)+IMG_HEADER);

     if(bytesread < (IMG_ROWS*IMG_COLS)+IMG_HEADER)
     {
         printf("Failure to read file in single request\n");
         exit(-1);
     }
     printf("bytesread = %d, bytes requested=%d\n", bytesread, (IMG_ROWS*IMG_COLS)+IMG_HEADER);
     strncpy((char *)header_str, (char *)oned_bigbuffer, (size_t)IMG_HEADER);
     printf("Image header: %s\n", header_str);
     close(fdin);


#ifdef TRANSFORM
     // parse out header and image data to find starting index of data


     // Put your rotation or other modifications of the data here
     //
     // E.g. transpose and pivot for rotation
     //
     // E.g. here as a starting point, I will create a negative transform
     for(idx=IMG_HEADER; idx < ((IMG_ROWS*IMG_COLS)+IMG_HEADER); idx++)
     {
         oned_bigbuffer[idx] = SATURATION - oned_bigbuffer[idx];
     }

#endif

     
     // Write back out the image
     sprintf(header_str, "#NEGATIVE");
     strncpy(&oned_bigbuffer[3], header_str, 9); // update the header comment to note NEGATIVE first!
     byteswritten = write(fdout, &oned_bigbuffer[0], (IMG_ROWS*IMG_COLS)+IMG_HEADER);

     if(byteswritten < (IMG_ROWS*IMG_COLS)+IMG_HEADER)
     {
         printf("Failure to write file in single request\n");
         exit(-1);
     }
     printf("byteswritten = %d, bytes requested=%d\n", byteswritten, (IMG_ROWS*IMG_COLS)+IMG_HEADER);
     close(fdout);
}
