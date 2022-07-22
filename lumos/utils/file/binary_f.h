#ifndef BINARY_F_H
#define BINART_F_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void bfget(FILE *fp, float *array, size_t size);
void bfput(FILE *fp, float *array, size_t size);

#ifdef __cplusplus
}
#endif

#endif