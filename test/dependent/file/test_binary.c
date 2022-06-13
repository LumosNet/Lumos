#include "binary_f.h"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    FILE *fp = fopen("./binary.w", "ab");
    size_t size = 20;
    float *array = calloc(size, sizeof(float));
    for (int i = 0; i < size; ++i){
        array[i] = i*0.1;
    }
    bfput(fp, array, size);
    fclose(fp);

    FILE *fg = fopen("./binary.w", "rb");
    float *n_array = calloc(size, sizeof(float));
    bfget(fg, n_array, size);
    for (int i = 0; i < size; ++i){
        if (n_array[i] != (float)(i*0.1)) {
            printf("[fail] Error in index %d\n", i);
            printf("[msg] Wrong number: %f\n", n_array[i]);
        } else{
            printf("[pass] binary file write/read correct\n");
        }
    }
    return 1;
}