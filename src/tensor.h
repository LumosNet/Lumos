#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

#include "lumos.h"
#include "list.h"

#ifdef  __cplusplus
extern "C" {
#endif

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