#ifndef CPU_GPU_H
#define CPU_GPU_H

#ifdef GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU
void fill_gpu(float *data, int len, float x, int offset);
void multy_gpu(float *data, int len, float x, int offset);
void add_gpu(float *data, int len, float x, int offset);

void matrix_add_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_subtract_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_multiply_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_divide_gpu(float *data_a, float *data_b, int num, float *space);

void saxpy_gpu(float *data_a, float *data_b, int num, float x, float *space);
void sum_channel_gpu(float *data, int h, int w, int c, float ALPHA, float *space);
#endif

#ifdef __cplusplus
}
#endif
#endif
