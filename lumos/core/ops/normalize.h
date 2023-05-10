#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "cpu.h"

#ifdef __cplusplus
extern "C"{
#endif

void normalize_mean(float *data, int h, int w, int c, float *mean);
void normalize_variance(float *data, int h, int w, int c, float *mean, float *variance);

void normalize_cpu(float *data, float *mean, float *variance, int h, int w, int c, float *space);

#ifdef  __cplusplus
}
#endif

#endif
