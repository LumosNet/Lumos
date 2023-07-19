#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "cpu.h"

#ifdef __cplusplus
extern "C"{
#endif

void normalize_mean(float *data, int h, int w, int c, float *mean);
void normalize_variance(float *data, int h, int w, int c, float *mean, float *variance);
void normalize_cpu(float *data, float *mean, float *variance, int h, int w, int c, float *space);

void gradient_normalize_mean(float *beta, float *variance, int num, float *mean_delta);
void gradient_normalize_variance(float *beta, float *input, float *n_delta, float *mean, float *variance, int h, int w, int c, float *variance_delta);
void gradient_normalize_cpu(float *input, float *mean, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta, float *space);
void gradient_normalize_layer(int h, int w, int c, float *l_delta, float *space);

#ifdef  __cplusplus
}
#endif

#endif
