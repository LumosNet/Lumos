#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

#include "lumos.h"
#include "list.h"

#ifdef  __cplusplus
extern "C" {
#endif

Tensor *tensor_d(int dim, int *size);
Tensor *tensor_x(int dim, int *size, float x);
Tensor *tensor_list(int dim, int *size, float *list);
Tensor *tensor_sparse(int dim, int *size, int **index, float *list, int n);
Tensor *tensor_copy(Tensor *ts);

void resize(Tensor *ts, int dim, int *size);
void tsprint(Tensor *ts);

float get_pixel(float *data, int dim, int *size, int *index);
void change_pixel(float *data, int dim, int *size, int *index, float x);

float min(float *data, int num);
float max(float *data, int num);
float mean(float *data, int num);

void add_x(float *data, int num, float x);
void mult_x(float *data, int num, float x);

void add(float *data_a, float *data_b, int num, float *space);
void subtract(float *data_a, float *data_b, int num, float *space);
void multiply(float *data_a, float *data_b, int num, float *space);
void divide(float *data_a, float *data_b, int num, float *space);

void saxpy(float *data_a, float *data_b, int num, float x, float *space);

#ifdef  __cplusplus
}
#endif

#endif