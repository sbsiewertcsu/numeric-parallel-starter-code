#include <stdio.h>
#include <time.h>
#include <omp.h>

void fred(void)
{
    printf("I am FRED\n");
}

void daisy(void)
{
    printf("I am DAISY\n");
}

void billy(void)
{
    printf("I am BILLY\n");
}

int main(int argc, char *argv[])
{

    printf("SINGLE:\n");

    #pragma omp parallel
    { 
       #pragma omp single
       { 
          #pragma omp task
             fred(); 
          #pragma omp task
             daisy(); 
          #pragma omp task
             billy(); 
       }  //end of single region
    } //end of parallel region

    printf("\nNOT SINGLE:\n");

    #pragma omp parallel
    { 
        #pragma omp task
           fred(); 
        #pragma omp task
           daisy(); 
        #pragma omp task
           billy(); 
    } //end of parallel region

}

