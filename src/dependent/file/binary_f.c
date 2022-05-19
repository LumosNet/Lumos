#include "binary_f.h"


// void write_as_binary(FILE *fp, float* array, size_t size)
// {
//     fwrite(array, sizeof(float), size, fp);
// }


// void read_part_as_bin(FILE *fp, float**array, size_t *arrsize, size_t length)
// {
//     fseek(fp, 0, SEEK_SET);
//     float *arr = (float*)malloc(sizeof(float)*length);
//     *array = arr;
//     int res = fread(arr, sizeof(float), length, fp);
//     *arrsize = res;
// }

// void read_all_as_binary(FILE *fp, float**array, size_t *arrsize)
// {
//     fseek(fp, 0, SEEK_END);
//     size_t end = ftell(fp)/sizeof(float);
//     read_part_as_bin(fp, array, arrsize, end);
// }

// void read_h2x_as_binary(FILE *fp, size_t length, float**array, size_t *arrsize)
// {
//     read_part_as_bin(fp, array, arrsize, length); 
// }

void bfget(FILE *fp, float *array, size_t size)
{
    fread(array, sizeof(float), size, fp);
}

void bfput(FILE *fp, float *array, size_t size)
{
    fwrite(array, sizeof(float), size, fp);
}