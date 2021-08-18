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

#endif