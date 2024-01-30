#ifndef SOFTMAX_GPU_H
#define SOFTMAX_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void softmax_gpu(float *data, int num, float *space, float *ALPHA);
void softmax_grident_gpu(float *data, int num, float *space, float *ALPHA);
void softmax_exp_sum_gpu(float *data, int num, float *workspace, float *space);

#ifdef __cplusplus
}
#endif
#endif
