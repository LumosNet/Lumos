#ifndef INCLUDE_H
#define INCLUDE_H

#include <stdio.h>
#include <stdlib.h>

// 第一维是列，第二维是行
typedef struct algebraic_space{
    float *data;
    int dim;
    int *size;
    int num;
} algebraic_space, AS, Array, array, Victor, victor;

typedef struct ASproxy{
    AS* (*create)(int, int*, float);
    AS* (*copy)(AS*);
    int (*get_lindex)(AS*, int*);
    int* (*get_mindex)(AS*, int);
    float (*get_pixel)(AS*, int*);
    void (*change_pixel)(AS*, int*, float);
    void (*replace_part)(AS*, AS*, int*);
    void (*resize)(AS*, int, int*);
    void (*slice)(AS*, float*, int*, int**);
    void (*merge)(AS*, AS*, int, int, float*);
    void (*del)(AS*);
    float (*get_sum)(AS*);
    float (*get_min)(AS*);
    float (*get_max)(AS*);
    float (*get_mean)(AS*);
    int (*get_num)(AS*, float);
    void (*saxpy)(AS*, AS*, float);
} proxy, Proxy, ASproxy, asproxy;

#endif