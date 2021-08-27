#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

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

tensor *copy(tensor *m);
int get_index(tensor *m, float x);
float get_pixel(tensor *m, int *index);
void resize(tensor *m, int dim, int *size);
void slice(tensor *m, float *workspace, int *dim_c, int **size_c);
void merge(tensor *m, tensor *n, int dim, int index, float *workspace);
float get_sum(tensor *m);
float get_min(tensor *m);
float get_max(tensor *m);
float get_mean(tensor *m);
void del(tensor *m);

#ifdef  __cplusplus
}
#endif

#endif