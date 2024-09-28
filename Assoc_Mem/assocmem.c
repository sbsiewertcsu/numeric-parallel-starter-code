#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Simple code to implement parallel content / associative memory search
//
// Algorithms for association:
//
// 1) Brute force search (sequential O(n) or parallel n/S where S is scaling to ((log2(n)))
// 2) Hash table (potentially O(1) for perfect hash, O(1) + m for imperfect where m is based on collisions)
//    * https://www.geeksforgeeks.org/associative-memory/
//    * bits in key sufficient to find larger key in smaller associative memory
//    * overflow collisions at key index can be stored in linked-list O(m) or binary tree O(log2(m))
// 3) Tree-structured (O(log2(n)) for binary tree)
// 4) CAM hardware (O(1))

typedef unsigned int pattern_t;
typedef unsigned long long int tablelen_t;

#define MAX_WORDS (1000000000ULL)
#define WORD_SIZE (sizeof(pattern_t))

pattern_t assoc_mem[MAX_WORDS];
int thread_count=2;

tablelen_t find_pattern(pattern_t pattern, pattern_t *mem, tablelen_t start_idx, tablelen_t size)
{
    tablelen_t idx, save_idx;
    int abort=0;

// Cancel loop when pattern found - cancel should work, but ignored
//
// http://jakascorner.com/blog/2016/08/omp-cancel.html
// * Note that cancellation can add overhead to OpenMP, so may be best just to compete loops
// * For this to work you must have OMP_CANCELLATION=true set as an environment variable
// * Check this with "printenv"
//
#pragma omp parallel for num_threads(thread_count)
    for(idx=start_idx; idx < size; idx++)
    {
       if(mem[idx] == pattern)
       {
           save_idx=idx;
           printf("thread %d found pattern %u\n", omp_get_thread_num(), pattern);

           // #pragma omp atomic
           #pragma omp critical
           {
               abort=1;
           }
           #pragma omp cancel for
       }

       if(abort)
       {
           printf("thread %d sees abort\n", omp_get_thread_num());
       }

       #pragma omp cancellation point for
    }

    printf("Search complete with index = %llu for size=%llu\n", save_idx, size);

    // if size is returned, the pattern was not found
    return save_idx; 
}


tablelen_t randomize_mem(pattern_t *mem, tablelen_t size)
{
    tablelen_t idx;
    srand(1011970);

#pragma omp parallel for num_threads(thread_count)
    for(idx=0; idx < size; idx++)
    {
        mem[idx]=(pattern_t)rand();
    }

	return mem[size-1];
}


tablelen_t sequence_mem(pattern_t *mem, tablelen_t size)
{
    tablelen_t idx;

#pragma omp parallel for num_threads(thread_count)
    for(idx=0; idx < size; idx++)
    {
        mem[idx]=(pattern_t)idx;
    }

	return size;
}


int main(int argc, char *argv[])
{
    pattern_t pattern = MAX_WORDS-1;
    tablelen_t size, pattern_idx;
    struct timespec start, stop;
    double fstart, fstop;

    // Process arguments to set pattern to find

    if(argc == 3) // key to find and # of threads
    {
        sscanf(argv[1], "%d", &thread_count);
        sscanf(argv[2], "%u", &pattern);
        printf("%d threads will search for %u\n", thread_count, pattern);
    }
    else if(argc == 2) // # of threads to find default key
    {
        sscanf(argv[1], "%d", &thread_count);
        printf("%d threads will search for %u\n", thread_count, pattern);
    }
    else
    {
        printf("Use: assocmem <# threads> <pattern>\n");
        printf("Will use defaults: %d threads will search for %u\n", thread_count, pattern);
    }

    printf("Start assoc memory init...\n");
    clock_gettime(CLOCK_MONOTONIC, &start); fstart=(double)start.tv_sec + ((double)start.tv_nsec/1000000000.0); 
    size = sequence_mem(assoc_mem, MAX_WORDS);
    clock_gettime(CLOCK_MONOTONIC, &stop); fstop=(double)stop.tv_sec + ((double)stop.tv_nsec/1000000000.0); 
    printf("Initialized assoc_mem with sequence up to %llu in %lf secs\n", size, (fstop-fstart));

    printf("Start assoc memory pattern search...\n");
    clock_gettime(CLOCK_MONOTONIC, &start); fstart=(double)start.tv_sec + ((double)start.tv_nsec/1000000000.0); 
    pattern_idx = find_pattern(pattern, assoc_mem, 0, MAX_WORDS);
    clock_gettime(CLOCK_MONOTONIC, &stop); fstop=(double)stop.tv_sec + ((double)stop.tv_nsec/1000000000.0); 
    if(pattern_idx < MAX_WORDS)
        printf("Found pattern=%u in assoc_mem at %llu in %lf secs\n", pattern, pattern_idx, (fstop-fstart));
    else
        printf("NOT FOUND: pattern=%u not in assoc_mem, searched in %lf secs\n", pattern, (fstop-fstart));

#if 0
    // This works, but pseudo-random initialization of a huge memory is very slow even in parallel

    printf("Start assoc memory RAND init...\n");
    clock_gettime(CLOCK_MONOTONIC, &start); fstart=(double)start.tv_sec + ((double)start.tv_nsec/1000000000.0); 
    pattern = randomize_mem(assoc_mem, MAX_WORDS);
    clock_gettime(CLOCK_MONOTONIC, &stop); fstop=(double)stop.tv_sec + ((double)stop.tv_nsec/1000000000.0); 
    printf("Initialized assoc_mem with sequence up to %llu in %lf secs\n", size, (fstop-fstart));

    printf("Start assoc memory pattern search...\n");
    clock_gettime(CLOCK_MONOTONIC, &start); fstart=(double)start.tv_sec + ((double)start.tv_nsec/1000000000.0); 
    pattern_idx = find_pattern(pattern, assoc_mem, 0, MAX_WORDS);
    clock_gettime(CLOCK_MONOTONIC, &stop); fstop=(double)stop.tv_sec + ((double)stop.tv_nsec/1000000000.0); 
    if(pattern_idx < MAX_WORDS)
        printf("Found pattern=%u in assoc_mem at %llu in %lf secs\n", pattern, pattern_idx, (fstop-fstart));
    else
        printf("NOT FOUND: pattern=%u not in assoc_mem, searched in %lf secs\n", pattern, (fstop-fstart));
#endif

}

