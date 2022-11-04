#ifndef POOLING_H
#define POOLING_H

#include "cpu.h"

#ifdef __cplusplus
extern "C"{
#endif

// ksize : 2
// stride : 2
// pad : 1

// 0 0 0 0 0 0
// 0 1 1 1 1 0     [i,j] [] []
// 0 1 1 1 1 0     []    [] []     --->    [x:i*stride, y:j*stride]
// 0 1 1 1 1 0     []    [] []
// 0 1 1 1 1 0
// 0 0 0 0 0 0

// 0,4 0,5
// 1,4 1,5     --->        [x-pad, y-pad]

// -1,2 -1,4
// 0,3 0,4

void avgpool(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space);
void maxpool(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index);

void avgpool_gradient(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n);
void maxpool_gradient(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n, int *index);

#ifdef  __cplusplus
}
#endif

#endif
