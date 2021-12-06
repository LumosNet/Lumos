#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

#include "lumos.h"
#include "list.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define SESSIONT

struct index_list
{
    int *index;
    struct index_list *next;
};

typedef struct index_list IndexList;

Tensor *tensor_d(int dim, int *size);
Tensor *tensor_x(int dim, int *size, float x);
Tensor *tensor_list(int dim, int *size, float *list);
Tensor *tensor_sparse(int dim, int *size, int **index, float *list, int n);
Tensor *tensor_copy(Tensor *ts);

void tsprint(Tensor *ts);

int ts_pixel_num(Tensor *ts, float x);
IndexList *ts_get_index(Tensor *ts, float x);
float ts_get_pixel(Tensor *ts, int *index);
void ts_change_pixel(Tensor *ts, int *index, float x);
void resize_ts(Tensor *ts, int dim, int *size);

float ts_sum(Tensor *ts);
float ts_min(Tensor *ts);
float ts_max(Tensor *ts);
float ts_mean(Tensor *ts);

void ts_add_x(Tensor *ts, float x);
void ts_mult_x(Tensor *ts, float x);

void ts_add(Tensor *ts_a, Tensor *ts_b);
void ts_subtract(Tensor *ts_a, Tensor *ts_b);
void ts_multiply(Tensor *ts_a, Tensor *ts_b);
void ts_divide(Tensor *ts_a, Tensor *ts_b);

void ts_saxpy(Tensor *ts_a, Tensor *ts_b, float x);

void free_tensor(Tensor *ts);

#ifdef NUMPY
void slice(Tensor *ts, float *workspace, int *dim_c, int **size_c);
void merge(Tensor *ts, Tensor *n, int dim, int index, float *workspace);

int __ergodic(Tensor *ts, int *index);
void __slice(Tensor *ts, int **sections, float *workspace, int dim);
void __slicing(Tensor *ts, int **sections, float *workspace, int dim_now, int offset_o, int *offset_a, int offset, int dim);
void __merging(Tensor *ts1, Tensor *ts2, int **sections, float *workspace, int dim_now, int offset_m, int *offset_n, int *offset_a, int offset, int *size, int dim);
#endif

#ifdef  __cplusplus
}
#endif

#endif