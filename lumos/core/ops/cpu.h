#ifndef CPU_H
#define CPU_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"{
#endif

// offset=1 为正常偏移
void fill_cpu(float *data, int len, float x, int offset);
void multy_cpu(float *data, int len, float x, int offset);
void add_cpu(float *data, int len, float x, int offset);

float min_cpu(float *data, int num);
float max_cpu(float *data, int num);
float sum_cpu(float *data, int num);
float mean_cpu(float *data, int num);

void one_hot_encoding(int n, int label, float *space);

void add(float *data_a, float *data_b, int num, float *space);
void subtract(float *data_a, float *data_b, int num, float *space);
void multiply(float *data_a, float *data_b, int num, float *space);
void divide(float *data_a, float *data_b, int num, float *space);

void saxpy(float *data_a, float *data_b, int num, float x, float *space);

void random(int range_l, int range_r, float scale, int num, float *space);

#ifdef __cplusplus
}
#endif

#endif