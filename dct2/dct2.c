// DCT and IDCT - listing 1
// Copyright (c) 2001 Emil Mikulic.
// http://unix4lyfe.org/dct/
//
// Feel free to do whatever you like with this code.
// Feel free to credit me.
//
// Adapted by Sam Siewert for Macroblock example to compare to OpenCV dct2
// and to Octave dct2.  Got rid of file format and just hard-coded example
// used in class and verifiable with Octave (MATLAB like tool for Linux) and
// or OpenCV cvDCT, both of which compute a DCT2 and inverse DCT2 (DCT3).
//
// Verified by comparing an example Macroblock graymap to DCT2 and iDCT2
// in Octave as follows:
//
// pkg load signal
//
// A=imread('exmblock.pgm')
// [m,n]=size(A)
// B=dct2(A)
// C=idct2(B)
//
// Restores original example Macroblock with no errors as expected.
//
// The original code posted on stackoverflow had an error in the inverse DCT2.
//
// You can install Octave on Ubuntu Linux with:
//
// sudo apt-add-repository -y ppa:picaso/octave
// sudo apt-get update
// sudo apt-get install octave
// sudo apt-get install liboctave-dev
//
// Here's what we seen in Octave:
// octave:14> A=imread('exmblock.pgm')
// A =
// 
//  101  100   94  102   97   91   88   83
//  101   99   98  103   93   93  107  110
//   98   97   97   97  103  101   94  100
//   97   98   99  100  103  105  101   96
//   99  100  104  104  100  107  109   89
//   99  101  105  105  116  113   87   58
//   94   69   66   66   79   70   40   26
//   59   30   33   33   32   37   45   41
//
// octave:15> [m,n]=size(A)
// m =  8
// n =  8
// octave:16> B=dct2(A)
// B =
// 
//  Columns 1 through 6:
// 
//   6.9525e+02   2.6010e+01  -1.6639e+01   2.5437e+01   7.5000e-01   7.6179e+00
//   1.2039e+02  -2.0490e+01   8.9012e+00  -2.5506e+01   1.1544e+00  -4.3287e+00
//  -1.0640e+02   1.1791e+01   1.7007e+01   1.4941e+00   1.3931e+01   4.5339e+00
//   5.6172e+01   2.3112e+01  -2.9161e+01   1.0977e+01  -9.6025e+00  -8.2810e-01
//  -2.6000e+01  -1.5603e+01   1.4815e+01  -8.4935e+00  -2.0000e+00   8.1834e+00
//  -8.9786e+00   2.3726e+01  -1.5388e+01   1.4887e+01   5.1144e+00  -1.0547e+01
//   5.1044e+00  -4.0175e+00  -9.5444e+00   1.8850e+00  -3.6055e+00  -3.9033e+00
//  -1.0798e+01   6.4487e+00   3.0031e+00   2.2585e+00   2.4347e+00  -5.1868e-01
//
// Columns 7 and 8:
//
//   2.2921e+00   1.9091e+00
//  -8.5313e+00  -5.7666e+00
//   7.0558e-01   2.5646e-01
//  -3.7586e+00  -1.2133e+00
//  -3.4306e+00  -1.0879e+00
//   4.5343e+00   1.8207e-01
//   7.4327e-01   4.4241e-01
//   1.6926e+00   6.0673e-02
//
// octave:17> C=idct2(B)
// C =
// 
// 101.000   100.000    94.000   102.000    97.000    91.000    88.000    83.000
// 101.000    99.000    98.000   103.000    93.000    93.000   107.000   110.000
//  98.000    97.000    97.000    97.000   103.000   101.000    94.000   100.000
//  97.000    98.000    99.000   100.000   103.000   105.000   101.000    96.000
//  99.000   100.000   104.000   104.000   100.000   107.000   109.000    89.000
//  99.000   101.000   105.000   105.000   116.000   113.000    87.000    58.000
//  94.000    69.000    66.000    66.000    79.000    70.000    40.000    26.000
//  59.000    30.000    33.000    33.000    32.000    37.000    45.000    41.000
//
// octave:18> 


#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// As documented for DCT2 in Wikipedia
//
// S[u,v] = 1/4 * C[u] * C[v] *
//   sum for x=0 to width-1 of
//   sum for y=0 to height-1 of
//     s[x,y] * cos( (2x+1)*u*PI / 2N ) * cos( (2y+1)*v*PI / 2N )
//
// C[u], C[v] = 1/sqrt(2) for u, v = 0
// otherwise, C[u], C[v] = 1
//
// S[u,v] ranges from -2^10 to 2^10
//

#define COEFFS(Cu,Cv,u,v) { \
	if (u == 0) Cu = 1.0 / sqrt(2.0); else Cu = 1.0; \
	if (v == 0) Cv = 1.0 / sqrt(2.0); else Cv = 1.0; \
	}

// http://planetmath.org/discretecosinetransform
//
// http://en.wikipedia.org/wiki/Discrete_cosine_transform
//
void dct(double macroblock[8][8], double dct2[8][8])
{
   int u,v,x,y;
   FILE * fp = fopen("dct2.csv", "w");
	
    for (v=0; v<8; v++)
    {
        for (u=0; u<8; u++)
        {
            double Cu, Cv, z = 0.0;

            COEFFS(Cu,Cv,u,v);

            for (y=0; y<8; y++)
            for (x=0; x<8; x++)
            {
                double s, q;

                s = macroblock[x][y];

                q = s * cos((double)(2*x+1) * (double)u * M_PI/16.0) *
                        cos((double)(2*y+1) * (double)v * M_PI/16.0);
                z += q;
            }

            dct2[v][u] = 0.25 * Cu * Cv * z;
            fprintf(fp, "\n %lf", dct2[v][u]);
        }
        fprintf(fp, "\n");

    }
}




void idct(double dct2[8][8], double idct2[8][8])
{
    int u,v,x,y;
    FILE * fp = fopen("idct2.csv", "w");

    for (y=0; y<8; y++)
    {
        for (x=0; x<8; x++)
        {
            double z = 0.0;

            for (v=0; v<8; v++)
            for (u=0; u<8; u++)
            {
                double S, q;
                double Cu, Cv;
		
                COEFFS(Cu,Cv,u,v);
                S = dct2[v][u];

                q = Cu * Cv * S *
                    cos((double)(2*x+1) * (double)u * M_PI/16.0) *
                    cos((double)(2*y+1) * (double)v * M_PI/16.0);

                    z += q;
            }
            z /= 4.0;
            idct2[x][y] = z;
            fprintf(fp, "\n %lf", idct2[x][y]);
        }
        fprintf(fp, "\n");
    }
}


// Test Example from class lecture notes, which can be compared to OpenCV
// or to Octave DCT2 and iDCT2 for verification.
//
// Note that this implementation is not efficient and is O(n^4) compared
// to best known algorithms that are O(nlog(n)) such as the AAN algorithm:
// Arai, Y.; Agui, T.; Nakajima, M. (November 1988). "A fast DCT-SQ scheme for
// images". IEICE Transactions 71 (11): 1095–1097. Which has similar performance
// to the Cooly-Tukey DFT for the discrete Fourier transform.
//
// This formulation is however much easier to understand.
//
int main()
{
    double Macroblock[8][8] = { {101, 100,  94, 102,  97,  91,  88,  83},
                                {101,  99,  98, 103,  93,  93, 107, 110},
                                { 98,  97,  97,  97, 103, 101,  94, 100},
                                { 97,  98,  99, 100, 103, 105, 101,  96},
                                { 99, 100, 104, 104, 100, 107, 109,  89},
                                { 99, 101, 105, 105, 116, 113,  87,  58},
                                { 94,  69,  66,  66,  79,  70,  40,  26},
                                { 59,  30,  33,  33,  32,  37,  45,  41} };
    double dct2[8][8];
    double idct2[8][8];

    // Trace output I added to these dump into a CSV file
    dct(Macroblock, dct2);
    idct(dct2, idct2);

    exit(0);
}
