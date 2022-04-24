#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "hash.h"


unsigned int n,m,k;
float p;

int* string_to_bytes(char* str){

    for (int i = 0; i < strlen(str); i++){
        str[i]
    }
}

int main(int argc, char **argv) {

    int c;
    opterr = 0;

//    Default values
    n = 100;
    p = 0.4;
    k = 2;
    m = 500;

//    DONE: Parse n (db size) and p (false positive percentage)
    while ((c = getopt (argc, argv, "np:")) != -1)
        switch (c)
        {
            case 'n':
                n = atoi(optarg);
                break;
            case 'b':
                p = atof(optarg);
                break;
            case '?':
                if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown option character `\\x%x'.\n",
                             optopt);
                return 1;
            default:
                abort ();
        }

//    TODO: Calculate m (bitfield size) and k (no of hash functions)

//    TODO: Handle actions like insert and lookup

//    TODO: Data value to hash key (string to bytes) convertor


//    TODO: Write 5 hash functions in hash.h and test implementation

//    TODO: Handle hash calls based on k value


    return 0;
}
