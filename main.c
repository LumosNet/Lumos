#include <stdio.h>
#include <stdlib.h>
#include "image.h"

int main(int argc, char **argv)
{
    float *a = malloc(2*sizeof(float));
    float *b = malloc(2*sizeof(float));
    float *c = malloc(2*sizeof(float));
    float *d = malloc(2*sizeof(float));
    a[0] = 0; a[1] = 0;
    b[0] = 1; b[1] = 1;
    c[0] = 0; c[1] = 1;
    d[0] = 1; d[1] = 0;
    save_image_data(a, 2, 1, 1, "./demo/xor/00.png");
    save_image_data(b, 2, 1, 1, "./demo/xor/11.png");
    save_image_data(c, 2, 1, 1, "./demo/xor/01.png");
    save_image_data(d, 2, 1, 1, "./demo/xor/10.png");
}