#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hash.h"

#define get_random rand()

#define M 10000000
#define max_str_length 11

typedef struct
{
    unsigned int bit:1;
}bits;

void printbitarray(bits *bitarray)
{
    for (int i=0; i<M; i++)
        if(bitarray[i].bit == 1)
            printf("%u ", i);
    printf("\n");
}

int main(int argc, char *argv[])
{

    FILE *fp_insert, *fp_lookup;
    int num_inserts, num_lookups, i;

    if (argc != 3)
    {
        printf("Usage: ./bloomfilter <inserts_filename> <lookups_filename>\n");
        exit(0);
    }

    char *insert_filename = argv[1];
    char *lookup_filename = argv[2];

    fp_insert = fopen(insert_filename, "r");
    fp_lookup = fopen(lookup_filename, "r");

    if (fp_insert == NULL)
    {
        printf("Not a valid insert file\n");
        exit(0);
    }

    if (fp_lookup == NULL)
    {
        printf("Not a valid lookup file\n");
        exit(0);
    }

    fscanf(fp_insert, "%d", &num_inserts);
    char **inserts = (char**) calloc(num_inserts, sizeof(char*));

    for (i=0; i<num_inserts; i++)
    {
        inserts[i] = (char*) calloc(max_str_length, sizeof(char));
        fscanf(fp_insert, "%s", inserts[i]);
    }

    fclose(fp_insert);

    fscanf(fp_lookup, "%d", &num_lookups);
    char **lookups = (char**) calloc(num_lookups, sizeof(char*));

    for (i=0; i<num_lookups; i++)
    {
        lookups[i] = (char*) calloc(max_str_length, sizeof(char));
        fscanf(fp_lookup, "%s", lookups[i]);
    }

    fclose(fp_lookup);

    srand(42);
    int seed1 = get_random;
    int seed2 = get_random;

    bits *bitvector = (bits*) calloc(M, sizeof(bits));
    for (i=0; i<M; i++)
        bitvector[i].bit = 0;

    for (i=0; i<num_inserts; i++){
        int p = hashmix(inserts[i], seed1, seed2) % M;
        int q = murmur(inserts[i], seed1) % M;
        int r = fnv1s(inserts[i]) % M;
        bitvector[p].bit = 1;
        bitvector[q].bit = 1;
        bitvector[r].bit = 1;

    }

    unsigned long long maybe = 0, notp = 0;
    int p, q, r;
    for (i=0; i<num_lookups; i++)
    {
        p = hashmix(lookups[i], seed1, seed2) % M;
        q = murmur(lookups[i], seed1) % M;
        r = fnv1s(lookups[i]) % M;

        if (bitvector[p].bit == 1 && bitvector[q].bit == 1 && bitvector[r].bit == 1)
            maybe++;
        else
            notp++;
    }

    printf("Maybe present: %llu\nNot present: %llu\n", maybe, notp);

    return 0;
}