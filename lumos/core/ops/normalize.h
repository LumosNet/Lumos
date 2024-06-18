#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "cpu.h"

#ifdef __cplusplus
extern "C"{
#endif

void normalize_mean(float *data, int h, int w, int c, int subdivision, float *mean);
void normalize_variance(float *data, int h, int w, int c, int subdivision, float *mean, float *variance);
void normalize_cpu(float *data, float *mean, float *variance, int h, int w, int c, float *space);

void gradient_normalize_mean(float *n_delta, float *variance, int h, int w, int c, float *mean_delta);
void gradient_normalize_variance(float *n_delta, float *input, float *mean, float *variance, int h, int w, int c, float *variance_delta);
void gradient_normalize_cpu(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta);

void update_scale(float *output, float *delta, int h, int w, int c, float rate, float *space);
void update_bias(float *delta, int h, int w, int c, float rate, float *space);

#ifdef  __cplusplus
}
#endif

#endif
