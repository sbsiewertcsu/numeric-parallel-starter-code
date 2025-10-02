#include <stdio.h>


#define NUM_1_HZ_ACCEL_VALUES (1801)
#define VALUES_PER_LINE (9)


double inputdata[NUM_1_HZ_ACCEL_VALUES];


int main(int argc, char *argv[])
{
    int rc;
    FILE *fin, *fout;

    if((fin = fopen(argv[1], "r")) == (FILE *)0)
    {
        printf("Error opening %s\n", argv[1]);
    }


    if((fout = fopen(argv[2], "w")) == (FILE *)0)
    {
        printf("Error opening %s\n", argv[2]);
    }


    fprintf(fout, "// Auto-generated from Excel CSV file with 15 digits of precision from Train design file\n");
    fprintf(fout, "//\n");
    fprintf(fout, "// From input file %s\n", argv[1]);
    fprintf(fout, "//\n");
    fprintf(fout, "//200 lines of 9 numbers + 1 last accerlation value\n");
    fprintf(fout, "//\n");
    fprintf(fout, "\n");
    fprintf(fout, "double DefaultProfile[] =\n");
    fprintf(fout, "{\n");

    for(int idx=0; idx < NUM_1_HZ_ACCEL_VALUES; idx++)
    {
        // read Excel CSV format file with inputdata values that are 15 digits of precision with a range
        // up to 5 leading digits, for a total field width of 20, so values should not be larger than 
        // 99,999.999999999999999
        //
        rc=fscanf(fin, "%lf\n", &inputdata[idx]);

        if((idx % 100) == 0)
            printf("a[%d]=%20.15lf\n", idx, inputdata[idx]);

        fprintf(fout, "%20.15lf, ", inputdata[idx]); 

        if((idx > 0) && ((idx+1) % VALUES_PER_LINE) == 0) fprintf(fout, "\n");
    }
    fprintf(fout, "\n};");

    fclose(fin);
    fclose(fout);
}
