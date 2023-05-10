#ifndef CPU_H
#define CPU_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C"{
#endif

// offset=1 为正常偏移
void fill_cpu(float *data, int len, float x, int offset);
void multy_cpu(float *data, int len, float x, int offset);
void add_cpu(float *data, int len, float x, int offset);

void min_cpu(float *data, int num, float *space);
void max_cpu(float *data, int num, float *space);
void sum_cpu(float *data, int num, float *space);
void mean_cpu(float *data, int num, float *space);
void variance_cpu(float *data, float mean, int num, float *space);

void matrix_add_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_subtract_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_multiply_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_divide_cpu(float *data_a, float *data_b, int num, float *space);

void saxpy_cpu(float *data_a, float *data_b, int num, float x, float *space);
void sum_channel_cpu(float *data, int h, int w, int c, float ALPHA, float *space);

void one_hot_encoding(int n, int label, float *space);

#ifdef __cplusplus
}
#endif

#endif