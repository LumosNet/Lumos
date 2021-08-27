#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

#include "list.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct session;

/* 行优先存储 */
typedef struct tensor{
    int dim;
    int *size;
    int num;
    float *data;
} tensor, Tensor;

struct index_list
{
    int *index;
    struct index_list *next;
};

typedef struct index_list index_list;


struct tensor_session{
    tensor* (*copy)();
    int (*get_index)();
    float (*get_pixel)();
    void (*resize)();
    void (*slice)();
    void (*merge)();
    float (*get_sum)();
    float (*get_min)();
    float (*get_max)();
    float (*get_mean)();
    void (*del)();
};

tensor *create_x(int dim, int *size, float x);
tensor *create_list(int dim, int *size, float *list);
Tensor *create_sparse(int dim, int *size, int **index, float *list);

tensor *copy(tensor *ts);
index_list *get_index(tensor *ts, float x);
float get_pixel(tensor *ts, int *index);
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