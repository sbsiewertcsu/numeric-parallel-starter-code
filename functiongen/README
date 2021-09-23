Some simple example C code utilities to deal with a train acceleration profile from an Excel spreadsheet

Linear interpolation can be used to look-up known acceleration values as given and to linearly interpolate to generate any
acceleration value as function of time.  This assumes that the acceleration function given is linear or at can be approximated
as piecewise linear.

For non-linear functions, the piecewise linear assumption will yield only approximate values of the original acceleration function, but
the advantage of this look-up and generate with interpoltion approach is that it works for any 1 Hz acceleration data, even if the
original function for acceleration is unknown - a function that can not be modeled with a math library function.

1) Use csvtostatic to convert and Excel plain CSV file with one column of acceleration values to a C static initializer

    sbsiewert@ecc-linux:~/code/functiongen$ make
    gcc -O3   -c timeprofiles.c
    gcc  -O3   -o timeprofiles timeprofiles.o
    gcc -O3   -c csvtostatic.c
    gcc  -O3   -o csvtostatic csvtostatic.o
    sbsiewert@ecc-linux:~/code/functiongen$ ls
    Ex3-Acceleration-Profile.csv  Ex6-Linear-Train.csv      Makefile     csvtostatic.c  ex3.h       ex4.h     ex4nonlin.h  ex6nonlin.h   timeprofiles.c
    Ex4-Acceleration-Profile.csv  Ex6-Non-Linear-Train.csv  csvtostatic  csvtostatic.o  ex3accel.h  ex4lin.h  ex6lin.h     timeprofiles  timeprofiles.o
    sbsiewert@ecc-linux:~/code/functiongen$ ./csvtostatic Ex4-Acceleration-Profile.csv ex4.h
    a[0]=0.00000000000000
    a[100]=0.08089391674863
    a[200]=0.15204348108239
    a[300]=0.20487812976367
    a[400]=0.23303348513394
    a[500]=0.23311799791334
    a[600]=0.20512148782702
    a[700]=0.15241636990484
    a[800]=0.08135141873633
    a[900]=0.00048700519193
    a[1000]=-0.08043607222249
    a[1100]=-0.15166994844470
    a[1200]=-0.20463390416124
    a[1300]=-0.23294798559398
    a[1400]=-0.23320152357431
    a[1500]=-0.20536397732078
    a[1600]=-0.15278861333307
    a[1700]=-0.08180857624833
    a[1800]=-0.00097400832168
    sbsiewert@ecc-linux:~/code/functiongen$ head ex4.h
    // Auto-generated from Excel CSV file with 15 digits of precision from Train design file
    //
    // From input file Ex4-Acceleration-Profile.csv
    //
    //200 lines of 9 numbers + 1 last accerlation value
    //

    double DefaultProfile[] =
    {
    0.00000000000000, 0.00082557972097, 0.00165114939568, 0.00247669897801, 0.00330221842206, 0.00412769768232, 0.00495312671375, 0.00577849547192, 0.00660379391317,
    sbsiewert@ecc-linux:~/code/functiongen$

2) Modify test program to use new exn.h file generated and verify that look-up and interpolation are working correcty by comparing to CSV

    sbsiewert@ecc-linux:~/code/functiongen$ ./timeprofiles
    argc=1, argv[0]=./timeprofiles
    Will use default time profile
    Number of values in profile = 1801 for 1801 expected
    A[0]=0.00000000000000
            A[0.00000000000000]=0.000000000
            A[0.10000000000000]=0.000082558
            A[0.20000000000000]=0.000165116
            A[0.30000000000000]=0.000247674
            A[0.40000000000000]=0.000330232
            A[0.50000000000000]=0.000412790
            A[0.60000000000000]=0.000495348
            A[0.70000000000000]=0.000577906
            A[0.80000000000000]=0.000660464
            A[0.90000000000000]=0.000743022
            A[1.00000000000000]=0.000825580
    A[1]=0.00082557972097
    A[1799]=-0.00179957512474
    A[1799.500000]=-0.00138679172321
    A[1800]=-0.00097400832168
    sbsiewert@ecc-linux:~/code/functiongen$ head Ex4-Acceleration-Profile.csv
    0.000000000000000
    0.000825579720969
    0.001651149395683
    0.002476698978011
    0.003302218422063
    0.004127697682319
    0.004953126713746
    0.005778495471925
    0.006603793913166
    0.007429011994639
    sbsiewert@ecc-linux:~/code/functiongen$


