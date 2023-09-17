// Sample directory contents reader from:
// https://stackoverflow.com/questions/25070751/navigating-through-files-using-dirent-h
//
// Could be used to automate finding all playing card file names rather than hard-coding the 52 unique
// file names in an array.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "cardlist.h"

char *pathcat(const char *str1, char *str2);

int main()
{
    struct dirent *dp;
    char *fullpath;
    const char *path="./../cards_3x4_ppm"; // Directory target
    DIR *dir = opendir(path); // Open the directory - dir contains a pointer to manage the dir

    printf("\nUse of readdir to find all card file names in a directory\n");
    while (dp=readdir(dir)) // if dp is null, there's no more content to read
    {

        // Simple name
        // printf("%s\n", dp->d_name);

        // Full path name
        fullpath = pathcat(path, dp->d_name);
        printf("%s\n", fullpath);
        free(fullpath);

    }

    closedir(dir); // close the handle (pointer)

    // Now here we demonstrate listing all hard-coded array card file names
    //
    printf("\n\nSimple use of a pre-filled array of card names rather than directory read\n");
    for(int idx=0; idx < 52; idx++)
        printf("%s\n", cardfilelist[idx]);

    return 0;
}

char *pathcat(const char *str1, char *str2)
{
    char *res;
    size_t strlen1 = strlen(str1);
    size_t strlen2 = strlen(str2);
    int i, j;

    res = malloc((strlen1+strlen2+1)*sizeof *res);
    strcpy(res, str1);

    for (i=strlen1, j=0; ((i<(strlen1+strlen2)) && (j<strlen2)); i++, j++)
        res[i] = str2[j];

    res[strlen1+strlen2] = '\0';
    return res;
}
