#ifndef CPU_GPU_H
#define CPU_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void fill_gpu(float *data, int len, float x, int offset);
void multy_gpu(float *data, int len, float x, int offset);
void add_gpu(float *data, int len, float x, int offset);

void min_gpu(float *data, int num, float *space);
void max_gpu(float *data, int num, float *space);
void sum_gpu(float *data, int num, float *space);
void mean_gpu(float *data, int num, float *space);
void variance_gpu(float *data, float *mean, int num, float *space);

void matrix_add_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_subtract_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_multiply_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_divide_gpu(float *data_a, float *data_b, int num, float *space);

void saxpy_gpu(float *data_a, float *data_b, int num, float x, float *space);
void sum_channel_gpu(float *data, int h, int w, int c, float ALPHA, float *space);

void exp_list_gpu(float *data, int num, float *space, float *ALPHA);

#ifdef __cplusplus
}
#endif
#endif
