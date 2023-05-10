#ifndef BIAS_H
#define BIAS_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void add_bias(float *origin, float *bias, int n, int size);
void scale_bias(float *origin, float *bias, int n, int size);

#ifdef __cplusplus
}
#endif

#endif