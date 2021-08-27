#ifndef ALGEBRAIC_SPACE_H
#define ALGEBRAIC_SPACE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "include.h"
#include "list.h"

// typedef struct tensorproxy{
//     tensor* (*create)(int, int*, float);
//     tensor* (*copy)(tensor*);
//     int (*get_lindex)(tensor*, int*);
//     int* (*get_mindex)(tensor*, int);
//     float (*get_pixel)(tensor*, int*);
//     void (*change_pixel)(tensor*, int*, float);
//     void (*replace_part)(tensor*, tensor*, int*);
//     void (*resize)(tensor*, int, int*);
//     void (*slice)(tensor*, float*, int*, int**);
//     void (*merge)(tensor*, tensor*, int, int, float*);
//     void (*del)(tensor*);
//     float (*get_sum)(tensor*);
//     float (*get_min)(tensor*);
//     float (*get_max)(tensor*);
//     float (*get_mean)(tensor*);
//     int (*get_num)(tensor*, float);
//     void (*saxpy)(tensor*, tensor*, float);
// } proxy, Proxy, tensorproxy, tensorproxy;

int __ergodic(tensor *m, int *index);
void __slice(tensor *m, int **sections, float *workspace, int dim);
void __slicing(tensor *m, int **sections, float *workspace, int dim_now, int offset_o, int *offset_a, int offset, int dim);
void __merging(tensor *m, tensor *n, int **sections, float *workspace, int dim_now, int offset_m, int *offset_n, int *offset_a, int offset, int *size, int dim);

tensor *create(int dim, int *size, float x);

tensor *list_to_tensor(int dim, int *size, float *list);
tensor *copy(tensor *m);

int get_lindex(tensor *m, int *index);
int *get_mindex(tensor *m, int index);

float get_pixel(tensor *m, int *index);
void change_pixel(tensor *m, int *index, float x);
void replace_part(tensor *m, tensor *n, int *index);

void resize(tensor *m, int dim, int *size);

void slice(tensor *m, float *workspace, int *dim_c, int **size_c);
void merge(tensor *m, tensor *n, int dim, int index, float *workspace);

void del(tensor *m);
float get_sum(tensor *m);
float get_min(tensor *m);
float get_max(tensor *m);
float get_mean(tensor *m);

void saxpy(tensor *mx, tensor *my, float x);
int get_num(tensor *m, float x);

// tensorproxy *init_tensorproxy();
#endif