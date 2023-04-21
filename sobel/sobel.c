// https://codereview.stackexchange.com/questions/262896/image-processing-sobel-edge-detection-in-c
//
// Adapted from above posted example
//
// Reads in BMP and writes it back out
//
// Has a Sobel transformation that can be called
//
// See Wikipedia for more information - https://en.wikipedia.org/wiki/Sobel_operator
//
// Build with "gcc sobel.c -o sobel -lm"
//
// Run with "./sobel" and at prompt enter "Bears" and it produces SobelTransformOut.bmp
//
// Sam Siewert
//
// 1) Combined pieces from post to create Sobel transform example using BMP test image
// 2) Clean up all warnings
// 3) Provided simple Makefile and tested
//

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_PATH 256
#define FILE_ROOT_PATH "./"

#define True true
#define False false


typedef struct RGB
{
    unsigned char R, G, B;
} RGB;

typedef struct BMPIMAGE
{
    char FILENAME[MAX_PATH];

    unsigned int XSIZE;
    unsigned int YSIZE;
    unsigned char FILLINGBYTE;
    unsigned char *IMAGE_DATA;
} BMPIMAGE;

typedef struct RGBIMAGE
{
    unsigned int XSIZE;
    unsigned int YSIZE;
    RGB *IMAGE_DATA;
} RGBIMAGE;

RGB *SobelEdge(const int xsize, const int ysize, const RGB * const image);

char bmp_read(unsigned char *image, const int xsize, const int ysize, const char *filename, const bool extension);
BMPIMAGE bmp_file_read(const char *filename, const bool extension);
int bmp_write(const unsigned char *image, const int xsize, const int ysize, const char *filename);
unsigned char *array_to_raw_image(const RGB* InputData, const int xsize, const int ysize);
RGB *raw_image_to_array(const unsigned char *image, const int xsize, const int ysize);
unsigned char bmp_filling_byte_calc(const unsigned int xsize);


void main(void)
{
    int rc;

    printf("BMP image file name:(ex:test): ");
    char *FilenameString;
    FilenameString = (char*)malloc( MAX_PATH * sizeof(char) );
    rc=scanf("%s",FilenameString);
    BMPIMAGE BMPImage1;


    BMPImage1 = bmp_file_read(FilenameString,false);

    free(FilenameString);

    printf("%s\n",BMPImage1.FILENAME);
    
    if(BMPImage1.IMAGE_DATA == NULL)
    {    
        printf("Image contents error!");
        exit(-1);
    }        
    
    RGBIMAGE RGBImage1;
    RGBImage1.XSIZE = BMPImage1.XSIZE;
    RGBImage1.YSIZE = BMPImage1.YSIZE;
    RGBImage1.IMAGE_DATA = (RGB*)malloc(RGBImage1.XSIZE * RGBImage1.YSIZE * sizeof(RGB));

    if (RGBImage1.IMAGE_DATA == NULL)
    {
        printf("Memory allocation error!");
        exit(-1);
    }

    RGBImage1.IMAGE_DATA = raw_image_to_array(BMPImage1.IMAGE_DATA, BMPImage1.XSIZE, BMPImage1.YSIZE);
    
    bmp_write(array_to_raw_image(SobelEdge(RGBImage1.XSIZE, RGBImage1.YSIZE, RGBImage1.IMAGE_DATA), RGBImage1.XSIZE, RGBImage1.YSIZE), BMPImage1.XSIZE, BMPImage1.YSIZE, "SobelTransformOutput");
    
};


RGB *SobelEdge(const int xsize, const int ysize, const RGB * const image)
{
    RGB *output;
    output = malloc(xsize * ysize * sizeof(RGB));
    if (output == NULL)
    {
        printf("Memory allocation error!");
        return NULL;
    }
    for(long long int i = 0; i < (xsize * ysize); i++)
    {
        if( (i < xsize) || ( (i % xsize) == 0) || ( ((i + 1) % xsize) == 0) || (i >= (xsize*(ysize-1))))
        {
            output[i].R = image[i].R;
            output[i].G = image[i].G;
            output[i].B = image[i].B;
        }
        else
        {
            int Gx = 0, Gy = 0;
            Gx = (
                image[i+xsize-1].R    * (-1) + image[i+xsize].R    * 0 + image[i+xsize+1].R    * 1 +
                image[i-1].R         * (-2) + image[i].R            * 0 + image[i+1].R         * 2 +
                image[i-xsize-1].R    * (-1) + image[i-xsize].R    * 0 + image[i-xsize+1].R    * 1
                );
            Gy = (
                image[i+xsize-1].R    * (-1) + image[i+xsize].R    * (-2) + image[i+xsize+1].R    * (-1) +
                image[i-1].R         *   0  + image[i].R            *   0  + image[i+1].R         *   0  +
                image[i-xsize-1].R    *   1  + image[i-xsize].R    *   2  + image[i-xsize+1].R    *   1
                );
            long double magnitude, direction;
            magnitude = sqrt(pow(Gx, 2) + pow(Gy, 2));
            if(Gx > 0)
            {
                direction = atan((long double)Gy / (long double)Gx) * ( 180 / M_PI) + (long double)90;
            }
            else if(Gx < 0)
            {
                direction = atan((long double)Gy / (long double)Gx) * ( 180 / M_PI) + (long double)270;
            }
            else if( (Gx == 0) && (Gy > 0) )
            {
                direction = 90;
            }
            else if( (Gx == 0) && (Gy < 0) )
            {
                direction = -90;
            }
            else if( (Gx == 0) && (Gy == 0) )
            {
                direction = 0;
            }

            output[i].R = (magnitude > 255)?255:(magnitude < 0)?0:(unsigned char)magnitude;

            Gx = (
                image[i+xsize-1].G    * (-1) + image[i+xsize].G    * 0 + image[i+xsize+1].G    * 1 +
                image[i-1].G         * (-2) + image[i].G            * 0 + image[i+1].G         * 2 +
                image[i-xsize-1].G    * (-1) + image[i-xsize].G    * 0 + image[i-xsize+1].G    * 1
                );
            Gy = (
                image[i+xsize-1].G    * (-1) + image[i+xsize].G    * (-2) + image[i+xsize+1].G    * (-1) +
                image[i-1].G         *   0  + image[i].G            *   0  + image[i+1].G         *   0  +
                image[i-xsize-1].G    *   1  + image[i-xsize].G    *   2  + image[i-xsize+1].G    *   1
                );
            magnitude = sqrt(pow(Gx, 2) + pow(Gy, 2));
            if(Gx > 0)
            {
                direction = atan((long double)Gy / (long double)Gx) * ( 180 / M_PI) + (long double)90;

            }
            else if(Gx < 0)
            {
                direction = atan((long double)Gy / (long double)Gx) * ( 180 / M_PI) + (long double)270;

            }
            else if( (Gx == 0) && (Gy > 0) )
            {
                direction = 90;
            }
            else if( (Gx == 0) && (Gy < 0) )
            {
                direction = -90;
            }
            else if( (Gx == 0) && (Gy == 0) )
            {
                direction = 0;
            }

            output[i].G = (magnitude > 255)?255:(magnitude < 0)?0:(unsigned char)magnitude;

            Gx = (
                image[i+xsize-1].B    * (-1) + image[i+xsize].B    * 0 + image[i+xsize+1].B    * 1 +
                image[i-1].B         * (-2) + image[i].B            * 0 + image[i+1].B         * 2 +
                image[i-xsize-1].B    * (-1) + image[i-xsize].B    * 0 + image[i-xsize+1].B    * 1
                );
            Gy = (
                image[i+xsize-1].B    * (-1) + image[i+xsize].B    * (-2) + image[i+xsize+1].B    * (-1) +
                image[i-1].B         *   0  + image[i].B            *   0  + image[i+1].B         *   0  +
                image[i-xsize-1].B    *   1  + image[i-xsize].B    *   2  + image[i-xsize+1].B    *   1
                );
            magnitude = sqrt(pow(Gx, 2) + pow(Gy, 2));
            if(Gx > 0)
            {
                direction = atan((long double)Gy / (long double)Gx) * ( 180 / M_PI) + (long double)90;

            }
            else if(Gx < 0)
            {
                direction = atan((long double)Gy / (long double)Gx) * ( 180 / M_PI) + (long double)270;

            }
            else if( (Gx == 0) && (Gy > 0) )
            {
                direction = 90;
            }
            else if( (Gx == 0) && (Gy < 0) )
            {
                direction = -90;
            }
            else if( (Gx == 0) && (Gy == 0) )
            {
                direction = 0;
            }

            output[i].B = (magnitude > 255)? 255: (magnitude < 0)?0:(unsigned char)magnitude;
        }
    }
    return output;
}


RGB *raw_image_to_array(const unsigned char *image, const int xsize, const int ysize)
{
    RGB *output;
    output = (RGB*)malloc(xsize * ysize * sizeof(RGB));
    if(output == NULL)
    {
        printf("Memory allocation error!");
        return NULL;
    }
    unsigned char FillingByte;
    FillingByte = bmp_filling_byte_calc(xsize);
    for(int y = 0;y<ysize;y++)
    {
        for(int x = 0;x<xsize;x++)
        {
            output[y*xsize + x].R =
            image[3*(y * xsize + x) + y * FillingByte + 2];
            output[y*xsize + x].G =
            image[3*(y * xsize + x) + y * FillingByte + 1];
            output[y*xsize + x].B =
            image[3*(y * xsize + x) + y * FillingByte + 0];
        }
    }
    return output;
}

//----bmp_read_x_size function----
unsigned long bmp_read_x_size(const char *filename, const bool extension)
{
    int rc;
    char fname_bmp[MAX_PATH];

    if(extension == false)
    {
        sprintf(fname_bmp, "%s.bmp", filename);
    }
    else
    {
        strcpy(fname_bmp,filename);
    }
    FILE *fp;
    fp = fopen(fname_bmp, "rb");
    if (fp == NULL)
    {
        printf("Fail to read file!\n");
        return -1;
    }
    unsigned char header[54];
    rc=fread(header, sizeof(unsigned char), 54, fp);
    unsigned long output;
    output = header[18] + (header[19] << 8) + ( header[20] << 16) + ( header[21] << 24);
    fclose(fp);
    return output;
}

//---- bmp_read_y_size function ----
unsigned long bmp_read_y_size(const char *filename, const bool extension)
{
    char fname_bmp[MAX_PATH];
    int rc;

    if(extension == false)
    {
        sprintf(fname_bmp, "%s.bmp", filename);
    }
    else
    {
        strcpy(fname_bmp,filename);
    }
    FILE *fp;
    fp = fopen(fname_bmp, "rb");
    if (fp == NULL)
    {
        printf("Fail to read file!\n");
        return -1;
    }
    unsigned char header[54];
    rc=fread(header, sizeof(unsigned char), 54, fp);
    unsigned long output;
    output= header[22] + (header[23] << 8) + ( header[24] << 16) + ( header[25] << 24);
    fclose(fp);
    return output;
}

//---- bmp_file_read function ----
char bmp_read(unsigned char *image, const int xsize, const int ysize, const char *filename, const bool extension)
{
    char fname_bmp[MAX_PATH];
    int rc;

    if(extension == false)
    {
        sprintf(fname_bmp, "%s.bmp", filename);
    }
    else
    {
        strcpy(fname_bmp,filename);
    }
    unsigned char filling_bytes;
    filling_bytes = bmp_filling_byte_calc(xsize);
    FILE *fp;
    fp = fopen(fname_bmp, "rb");
    if (fp==NULL)
    {
        printf("Fail to read file!\n");
        return -1;
    }
    unsigned char header[54];
    rc=fread(header, sizeof(unsigned char), 54, fp);
    rc=fread(image, sizeof(unsigned char), (size_t)(long)(xsize * 3 + filling_bytes)*ysize, fp);
    fclose(fp);
    return 0;
}

BMPIMAGE bmp_file_read(const char *filename, const bool extension)
{
    BMPIMAGE output;
    stpcpy(output.FILENAME, "");
    output.XSIZE = 0;
    output.YSIZE = 0;
    output.IMAGE_DATA = NULL;
    if(filename == NULL)
    {
        printf("Path is null\n");
        return output;
    }
    char fname_bmp[MAX_PATH];
    if(extension == false)
    {
        sprintf(fname_bmp, "%s.bmp", filename);
    }
    else
    {
        strcpy(fname_bmp,filename);
    }
    FILE *fp;
    fp = fopen(fname_bmp, "rb");
    if (fp == NULL)
    {
        printf("Fail to read file!\n");
        return output;
    }
    stpcpy(output.FILENAME, fname_bmp);
    output.XSIZE = (unsigned int)bmp_read_x_size(output.FILENAME,true);
    output.YSIZE = (unsigned int)bmp_read_y_size(output.FILENAME,true);
    if( (output.XSIZE == -1) || (output.YSIZE == -1) )
    {
        printf("Fail to fetch information of image!");
        return output;
    }
    else
    {
        printf("Width of the input image: %d\n",output.XSIZE);
        printf("Height of the input image: %d\n",output.YSIZE);
        printf("Size of the input image(Byte): %lu\n",(size_t)output.XSIZE * output.YSIZE * 3);
        output.FILLINGBYTE = bmp_filling_byte_calc(output.XSIZE);
        output.IMAGE_DATA = (unsigned char*)malloc((output.XSIZE * 3 + output.FILLINGBYTE) * output.YSIZE * sizeof(unsigned char));
        if (output.IMAGE_DATA == NULL)
        {
            printf("Memory allocation error!");
            return output;
        }
        else
        {
            for(int i = 0; i < ((output.XSIZE * 3 + output.FILLINGBYTE) * output.YSIZE);i++)
            {
                output.IMAGE_DATA[i] = 255;
            }
            bmp_read(output.IMAGE_DATA, output.XSIZE, output.YSIZE, output.FILENAME, true);
        }
    }
    return output;
}

//----bmp_write function----
int bmp_write(const unsigned char *image, const int xsize, const int ysize, const char *filename)
{
    unsigned char FillingByte;
    FillingByte = bmp_filling_byte_calc(xsize);
    unsigned char header[54] =
    {
    0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
    54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0
    };
    unsigned long file_size = (long)xsize * (long)ysize * 3 + 54;
    unsigned long width, height;
    char fname_bmp[MAX_PATH];
    header[2] = (unsigned char)(file_size &0x000000ff);
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    width = xsize;
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) &0x000000ff;
    header[20] = (width >> 16) &0x000000ff;
    header[21] = (width >> 24) &0x000000ff;

    height = ysize;
    header[22] = height &0x000000ff;
    header[23] = (height >> 8) &0x000000ff;
    header[24] = (height >> 16) &0x000000ff;
    header[25] = (height >> 24) &0x000000ff;
    sprintf(fname_bmp, "%s.bmp", filename);
    FILE *fp;
    if (!(fp = fopen(fname_bmp, "wb")))
    {
        return -1;
    }
    fwrite(header, sizeof(unsigned char), 54, fp);
    fwrite(image, sizeof(unsigned char), (size_t)(long)(xsize * 3 + FillingByte)*ysize, fp);
    fclose(fp);
    return 0;
}

unsigned char *array_to_raw_image(const RGB* InputData, const int xsize, const int ysize)
{
    unsigned char FillingByte;
    FillingByte = bmp_filling_byte_calc(xsize);
    unsigned char *output;
    output = (unsigned char*)malloc((xsize * 3 + FillingByte) * ysize * sizeof(unsigned char));
    if(output == NULL)
    {
        printf("Memory allocation error!");
        return NULL;
    }
    for(int y = 0;y < ysize;y++)
    {
        for(int x = 0;x < (xsize * 3 + FillingByte);x++)
        {
            output[y * (xsize * 3 + FillingByte) + x] = 0;
        }
    }
    for(int y = 0;y<ysize;y++)
    {
        for(int x = 0;x<xsize;x++)
        {
            output[3*(y * xsize + x) + y * FillingByte + 2]
            = InputData[y*xsize + x].R;
            output[3*(y * xsize + x) + y * FillingByte + 1]
            = InputData[y*xsize + x].G;
            output[3*(y * xsize + x) + y * FillingByte + 0]
            = InputData[y*xsize + x].B;
        }
    }
    return output;
}

unsigned char bmp_filling_byte_calc(const unsigned int xsize)
{
    unsigned char filling_bytes;
    filling_bytes = ( xsize % 4);
    return filling_bytes;
}
