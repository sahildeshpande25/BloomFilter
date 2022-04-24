//
// Created by Akul  Santhosh on 4/21/22.
//

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "hash.h"

#define rand_a rand()
#define rand_b rand()


int main(){
    char* a = "abc";
    int seed1 = rand_a;
    int seed2 = rand_b;
    int p = hashmix(a, seed1, seed2) % 15;
    int q = murmur(a, seed1) % 15;
    int r = fnv1s(a) % 15;
    printf("%d %d %d", p, q, r);
    puts("");
}

