#ifndef BINARY_F_H
#define BINART_F_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// void write_as_binary(FILE *fp, float* array, size_t size);

// void read_all_as_binary(FILE *fp, float**array, size_t *arrsize);
// void read_h2x_as_binary(FILE *fp, size_t length, float**array, \
//                         size_t *arrsize);

void bfget(FILE *fp, float *array, size_t size);
void bfput(FILE *fp, float *array, size_t size);

#ifdef __cplusplus
}
#endif

#endif