#ifndef BIAS_GPU_H
#define BIAS_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void add_bias_gpu(float *origin, float *bias, int n, int size);
void scale_bias_gpu(float *origin, float *bias, int n, int size);

#ifdef __cplusplus
}
#endif
#endif
