#include <string.h>

int* string_to_bytes(char* str)
{
    int n = strlen(str);
    static int bytes[100];
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

    return bytes;
}


int fnv1s(char* str)
{

    int FNVPRIME = 0x01000193;
    int FNVINIT = 0x811c9dc5;

    int *p = string_to_bytes(str);
    int hash = FNVINIT;
    for (int i = 0; i < strlen(str)*8; i++)
    {
        hash *= FNVPRIME;
        hash ^= p[i];
    }

    return abs(hash);
}

int hashmix(char* str, int a, int b)
{
    int *p = string_to_bytes(str);
    int c = p[0];

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
        c ^= p[i];
    }

    return abs(c);
}

unsigned int murmur (char* key, int seed)
{
    const unsigned int m = 0x5bd1e995;
    const int r = 24;

    int len = strlen(key);
    unsigned int h = seed ^ len;

    const unsigned char * data = (const unsigned char *)key;

    while(len >= 4)
    {
        unsigned int k = *(unsigned int *)data;

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        len -= 4;
    }

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

unsigned long djb2(char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c;

    return hash;
}