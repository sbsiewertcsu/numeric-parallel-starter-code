#include <stdio.h>


#define NUM_1_HZ_ACCEL_VALUES (1801)
#define VALUES_PER_LINE (9)


double accel[NUM_1_HZ_ACCEL_VALUES];


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
        // read Excel CSV format file with accel values that are 15 digits < 9.99999999999999 m/sec
        rc=fscanf(fin, "%lf\n", &accel[idx]);

        if((idx % 100) == 0) printf("a[%d]=%015.14lf\n", idx, accel[idx]);
        fprintf(fout, "%015.14lf, ", accel[idx]); 

        if((idx > 0) && ((idx+1) % VALUES_PER_LINE) == 0) fprintf(fout, "\n");
    }
    fprintf(fout, "\n};");

    fclose(fin);
    fclose(fout);
}
