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

    for (int i=0; i<3; i++){
        int p = hashmix(a[i], seed1, seed2) % M;
        int q = murmur(a[i], seed1) % M;
        int r = fnv1s(a[i]) % M;
//        printf("%s %d %d %d \n", a[i], p, q, r);
        bitvector[p].bit = 1;
        bitvector[q].bit = 1;
        bitvector[r].bit = 1;

    }

    char *test[5] = {"cpp", "sahil", "akul", "random", "random2"};
    for (int i=0; i<5; i++)
    {
        int p = hashmix(test[i], seed1, seed2) % M;
        int q = murmur(test[i], seed1) % M;
        int r = fnv1s(test[i]) % M;

        printf("%d %d %d\n", p, q, r);
        if (bitvector[p].bit == 1 && bitvector[q].bit == 1 && bitvector[r].bit == 1)
            printf("%s may be present\n", test[i]);
        else
            printf("%s is not present\n", test[i]);
    }

    puts("");
}

