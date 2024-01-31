#ifndef NORMALIZE_GPU_H
#define NORMALIZE_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void normalize_mean_gpu(float *data, int h, int w, int c, float *mean);
void normalize_variance_gpu(float *data, int h, int w, int c, float *mean, float *variance);
void normalize_gpu(float *data, float *mean, float *variance, int h, int w, int c, float *space);

void gradient_normalize_mean_gpu(float *beta, float *variance, int num, float *mean_delta);
void gradient_normalize_variance_gpu(float *beta, float *input, float *n_delta, float *mean, float *variance, int h, int w, int c, float *variance_delta);
void gradient_normalize_gpu(float *input, float *mean, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta, float *space);
void gradient_normalize_layer_gpu(int h, int w, int c, float *l_delta, float *space);

#ifdef __cplusplus
}
#endif
#endif
