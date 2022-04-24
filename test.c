#include <stdio.h>
#include <stdlib.h>
#include "hash.h"

#define rand_a rand()
#define rand_b rand()

#define M 100
#define N 10

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

int main(){
    char *a[3] = {"hello", "world", "cpp"};
    int seed1 = rand_a;
    int seed2 = rand_b;

    bits bitvector[M];
    for (int i=0; i<M; i++)
        bitvector[i].bit = 0;
    printbitarray(bitvector);

    for (int i=0; i<3; i++){
        int p = hashmix(a[i], seed1, seed2) % 15;
        int q = murmur(a[i], seed1) % 15;
        int r = fnv1s(a[i]) % 15;
        printf("%s %d %d %d \n", a[i], p, q, r);
        bitvector[p].bit = 1;
        bitvector[q].bit = 1;
        bitvector[r].bit = 1;

        printbitarray(bitvector);
    }

    puts("");
}

