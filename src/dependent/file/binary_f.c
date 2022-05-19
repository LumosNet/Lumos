#include "binary_f.h"

void bfget(FILE *fp, float *array, size_t size)
{
    fread(array, sizeof(float), size, fp);
}

void bfput(FILE *fp, float *array, size_t size)
{
    fwrite(array, sizeof(float), size, fp);
}