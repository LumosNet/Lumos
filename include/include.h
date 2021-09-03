#ifndef INCLUDE_H
#define INCLUDE_H

#include <stdio.h>
#include <stdlib.h>

typedef struct session;

/* 行优先存储 */
typedef struct tensor{
    int dim;
    int *size;
    int num;
    float *data;
    struct session *ptr;
} tensor, Tensor;

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
};

tensor *tensor_x(int dim, int *size, float x);
tensor *tensor_list(int dim, int *size, float *list);
Tensor *tensor_sparse(int dim, int *size, int **index, float *list);

#endif