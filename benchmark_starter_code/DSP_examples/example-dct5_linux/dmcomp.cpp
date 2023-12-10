/*
 *
 *  Example by Sam Siewert for use with PPM images to compress with a 
 *  a process similar to MPEG, to provide a simple example.
 *
 *  PART 1: Simple conversion of image to DCT and back with inverse DCT
 *          Shows 8x8 ROI DCT for 320x240 image (40 x 30 8x8 macroblocks)
 *
 *  Based on numerous code snippets from stackoverflow.com
 *
 */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    IplImage *image;                  // basic image pointer
    IplImage *b,*g,*r;                // blue, green, red only images
    IplImage *bf, *gf, *rf;           // float versions
    IplImage *b_dct,*g_dct,*r_dct;    // DCT converted float versions
    IplImage *b_idct,*g_idct,*r_idct; // DCT converted float versions
    int i, j;

    // Read in command line supplied file argument for PPM, JPG, etc.
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    // Read in the source image file
    image = cvLoadImage(argv[1]);

    // Allocate each color band image and set pointer
    b=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
    g=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
    r=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);

    // Split read in image into the three color bands
    cvSplit(image, b, g, r, NULL);

    // Allocate each float color band image and set pointer, then convert
    bf = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    cvConvert(b, bf);
    gf = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    cvConvert(g, gf);
    rf = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    cvConvert(r, rf);

    // Allocate each float DCT, iDCT color band image and set pointer
    b_dct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    b_idct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    g_dct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    g_idct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    r_dct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    r_idct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);

    // Transform B with DCT in 8x8 sub-regions
    for(i=0; i < 320; i=i+8)
    {
        for(j=0; j < 240; j=j+8)
        {
            cvSetImageROI(bf, cvRect(i,j,8,8));
            cvSetImageROI(b_dct, cvRect(i,j,8,8));
            cvDCT(bf, b_dct, 0);
        }
    }
    cvResetImageROI(bf);
    cvResetImageROI(b_dct);

    // Transform G with DCT in 8x8 sub-regions
    for(i=0; i < 320; i=i+8)
    {
        for(j=0; j < 240; j=j+8)
        {
            cvSetImageROI(gf, cvRect(i,j,8,8));
            cvSetImageROI(g_dct, cvRect(i,j,8,8));
            cvDCT(gf, g_dct, 0);
        }
    }
    cvResetImageROI(gf);
    cvResetImageROI(g_dct);

    // Transform R with DCT in 8x8 sub-regions
    for(i=0; i < 320; i=i+8)
    {
        for(j=0; j < 240; j=j+8)
        {
            cvSetImageROI(rf, cvRect(i,j,8,8));
            cvSetImageROI(r_dct, cvRect(i,j,8,8));
            cvDCT(rf, r_dct, 0);
        }
    }
    cvResetImageROI(rf);
    cvResetImageROI(r_dct);


    // Transform with DCT in 8x8 sub-regions
    for(i=0; i < 320; i=i+8)
    {
        for(j=0; j < 240; j=j+8)
        {
            cvSetImageROI(b_dct, cvRect(i,j,8,8));
            cvSetImageROI(b_idct, cvRect(i,j,8,8));
            cvDCT(b_dct, b_idct, DCT_INVERSE);
        }
    }
    cvResetImageROI(b_dct);
    cvResetImageROI(b_idct);

    // Transform with DCT in 8x8 sub-regions
    for(i=0; i < 320; i=i+8)
    {
        for(j=0; j < 240; j=j+8)
        {
            cvSetImageROI(g_dct, cvRect(i,j,8,8));
            cvSetImageROI(g_idct, cvRect(i,j,8,8));
            cvDCT(g_dct, g_idct, DCT_INVERSE);
        }
    }
    cvResetImageROI(g_dct);
    cvResetImageROI(g_idct);

    // Transform with DCT in 8x8 sub-regions
    for(i=0; i < 320; i=i+8)
    {
        for(j=0; j < 240; j=j+8)
        {
            cvSetImageROI(r_dct, cvRect(i,j,8,8));
            cvSetImageROI(r_idct, cvRect(i,j,8,8));
            cvDCT(r_dct, r_idct, DCT_INVERSE);
        }
    }
    cvResetImageROI(r_dct);
    cvResetImageROI(r_idct);


    // Create a window for display.
    namedWindow("Input Image Display window", CV_WINDOW_AUTOSIZE );

    // Show original image
    cvShowImage("Input Image Display window", image);

    // Wait for a keystroke in the window
    waitKey(0);

    // convert float DCT color band to simple image
    cvConvert(b_dct, b);
    cvConvert(g_dct, g);
    cvConvert(r_dct, r);

    namedWindow("BLUE Display window", CV_WINDOW_AUTOSIZE );
    cvShowImage("BLUE Display window", b);
    waitKey(0);
    namedWindow("GREEN Display window", CV_WINDOW_AUTOSIZE );
    cvShowImage("GREEN Display window", g);
    waitKey(0);
    namedWindow("RED Display window", CV_WINDOW_AUTOSIZE );
    cvShowImage("RED Display window", r);
    waitKey(0);

    // convert float iDCT b color band to simple image
    cvConvert(b_idct, b);
    cvConvert(g_idct, g);
    cvConvert(r_idct, r);

    // Split read in image into the three color bands
    cvMerge(b, g, r, NULL, image);

    namedWindow("RECOVERED Display window", CV_WINDOW_AUTOSIZE );
    cvShowImage("RECOVERED Display window", image);
    // Wait for a keystroke in the window
    waitKey(0);

    return 0;
};
