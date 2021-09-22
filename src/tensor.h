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

typedef struct index_list index_list;

tensor *tensor_x(int dim, int *size, float x);
tensor *tensor_list(int dim, int *size, float *list);
Tensor *tensor_sparse(int dim, int *size, int **index, float *list, int n);

void tsprint(tensor *ts);

tensor *copy(tensor *ts);
void tsprint(tensor *ts);
int pixel_num(tensor *ts, float x);
index_list *get_index(tensor *ts, float x);
float get_pixel(tensor *ts, int *index);
void change_pixel(tensor *ts, int *index, float x);
void resize(tensor *ts, int dim, int *size);
void slice(tensor *ts, float *workspace, int *dim_c, int **size_c);
void merge(tensor *ts, tensor *n, int dim, int index, float *workspace);
float get_sum(tensor *ts);
float get_min(tensor *ts);
float get_max(tensor *ts);
float get_mean(tensor *ts);
void del(tensor *ts);

int __ergodic(tensor *ts, int *index);
void __slice(tensor *ts, int **sections, float *workspace, int dim);
void __slicing(tensor *ts, int **sections, float *workspace, int dim_now, int offset_o, int *offset_a, int offset, int dim);
void __merging(tensor *ts1, tensor *ts2, int **sections, float *workspace, int dim_now, int offset_m, int *offset_n, int *offset_a, int offset, int *size, int dim);

#ifdef  __cplusplus
}
#endif

#endif