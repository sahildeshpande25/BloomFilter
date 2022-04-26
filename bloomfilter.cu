#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define get_random rand()

#define M 100000000
#define max_str_length 8

using namespace std;

__device__ int strlen(char *str)
{
    int c;
    for (c = 0; str[c] != '\0'; c++);
    return c;
}


__device__ void string_to_bytes(char* str, int *bytes)
{
    int n = strlen(str);
    unsigned char byte;
    int i, j, c = 0;

    for (i = 0; i < n ; i++)
    {
        for (j = 7; j >= 0; j--)
        {
            byte = (str[i] >> j) & 1;
            bytes[c++] = byte;
        }
    }
}


__device__ int fnv1s(char* str)
{

    int FNVPRIME = 0x01000193;
    int FNVINIT = 0x811c9dc5;

    int bytes[100];
    string_to_bytes(str, bytes);
    int hash = FNVINIT;
    for (int i = 0; i < strlen(str)*8; i++)
    {
        hash *= FNVPRIME;
        hash ^= bytes[i];
    }

    return abs(hash);
}


__device__ int hashmix(char* str, int a, int b)
{
    int bytes[100];
    string_to_bytes(str, bytes);
    int c = bytes[0];

    for (int i = 1; i < strlen(str)*8; i++)
    {
        a -= (b + c);  a ^= (c >> 13);
        b -= (c + a);  b ^= (a << 8);
        c -= (a + b);  c ^= (b >> 13);
        a -= (b + c);  a ^= (c >> 12);
        b -= (c + a);  b ^= (a << 16);
        c -= (a + b);  c ^= (b >> 5);
        a -= (b + c);  a ^= (c >> 3);
        b -= (c + a);  b ^= (a << 10);
        c -= (a + b);  c ^= (b >> 15);
        c ^= bytes[i];
    }

    return abs(c);
}


__device__ int murmur (char* key, int seed)
{
    int m = 0x5bd1e995;

    int len = strlen(key);
    int h = seed ^ len;

    char * data = key;

    switch(len)
    {
        case 4: h ^= data[3] << 24;
        case 3: h ^= data[2] << 16;
        case 2: h ^= data[1] << 8;
        case 1: h ^= data[0];
            h *= m;
    };

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}


__device__ unsigned long djb2(char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}


void print_bit_array(char *bitarray)
{
    for (int i=0; i<M; i++)
        if(bitarray[i] == 1)
            printf("%d ", i);
    printf("\n");
}

__global__ void insert_parallel(char *, char *, int, int, int);

__global__ void lookup_parallel(char *, char *, int, int, int, char *);


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
    char *inserts = (char *) malloc(num_inserts*max_str_length*sizeof(char));

    for (i=0; i<num_inserts; i++)
    {
        fscanf(fp_insert, "%s", &inserts[i*max_str_length]);
    }
    fclose(fp_insert);

    fscanf(fp_lookup, "%d", &num_lookups);
    char *lookups = (char *) malloc(num_lookups*max_str_length*sizeof(char));

    for (i=0; i<num_lookups; i++)
    {
        fscanf(fp_lookup, "%s", &lookups[i*max_str_length]);
    }
    fclose(fp_lookup);

    srand(42);
    int seed1 = get_random;
    int seed2 = get_random;

    char *c_inserts = NULL;
    if (cudaMalloc((void**)&c_inserts, num_inserts*max_str_length*sizeof(char)) != cudaSuccess )
    {
        printf("Error while allocating memory for insert array");
        exit(1);
    }

    char *c_bits = NULL;
    if (cudaMalloc((void**)&c_bits, M*sizeof(char)) != cudaSuccess )
    {
        printf("Error while allocating memory for bit array");
        exit(1);
    }

    cudaMemset(c_bits, 0, M*sizeof(char));

    char *c_maybe = NULL;
    if (cudaMalloc((void**)&c_maybe, num_lookups*sizeof(char)) != cudaSuccess )
    {
        printf("Error while allocating memory for maybe array");
        exit(1);
    }

    char *c_lookups = NULL;
    if (cudaMalloc((void**)&c_lookups, num_lookups*max_str_length*sizeof(char)) != cudaSuccess )
    {
        printf("Error while allocating memory for insert array");
        exit(1);
    }

    if (cudaMemcpy(c_inserts, inserts, num_inserts*max_str_length*sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("Error while copying to device from Host for insert array");
        exit(1);
    }

    if (cudaMemcpy(c_lookups, lookups, num_lookups*max_str_length*sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("Error while copying to device from Host for lookup array");
        exit(1);
    }

    int no_of_block = (int)(num_inserts/64) + 1;
    insert_parallel<<<no_of_block, 64>>>(c_inserts, c_bits, num_inserts, seed1, seed2);
    cudaDeviceSynchronize();

    int no_of_block_lookups = (int)(num_lookups/64) + 1;
    lookup_parallel<<<no_of_block_lookups, 64>>>(c_lookups, c_bits, num_lookups, seed1, seed2, c_maybe);
    cudaDeviceSynchronize();

    char *maybe = (char *) malloc(num_lookups*sizeof(char));
    cudaMemcpy(maybe, c_maybe, num_lookups*sizeof(char), cudaMemcpyDeviceToHost);

    unsigned long long count_maybe = 0, count_notp = 0;
    for (int i = 0; i < num_lookups; i++)
    {
        if (maybe[i]) count_maybe++;
        else count_notp++;
    }

    printf("\nGPU Version (Parallel)\n");
    printf("\nNumber of inserts: %10d\n", num_inserts);
    printf("Number of lookups: %10d\n", num_lookups);
    printf("\nLookup Summary:\n");
    printf("Number of elements that MAY BE present:  %10d\n", count_maybe);
    printf("Number of elements that are NOT present: %10d\n\n", count_notp);

    return 0;
}

__global__ void insert_parallel(char *inserts, char *bits, int size, int seed1, int seed2){
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size)
    {
        int p = hashmix(&inserts[i*max_str_length], seed1, seed2) % M;
        int w = djb2(&inserts[i*max_str_length]) % M;
        int r = fnv1s(&inserts[i*max_str_length]) % M;
        bits[p] = 1;
        bits[w] = 1;
        bits[r] = 1;
    }
}

__global__ void lookup_parallel(char *inserts, char *bits, int size, int seed1, int seed2, char *maybe){
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size)
    {

        int p = hashmix(&inserts[i*max_str_length], seed1, seed2) % M;
        int w = djb2(&inserts[i*max_str_length]) % M;
        int r = fnv1s(&inserts[i*max_str_length]) % M;
        if (bits[p] == 1 && bits[w] == 1 && bits[r] == 1) maybe[i] = 1;
        else maybe[i] = 0;
    }
}