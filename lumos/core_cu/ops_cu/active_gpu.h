#ifndef ACTIVE_GPU_H
#define ACTIVE_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "active.h"
#include "gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef float   (*activate_gpu)(float);
typedef float   (*gradient_gpu)(float);
typedef activate_gpu ActivateGpu;
typedef gradient_gpu GradientGpu;

ActivateGpu load_activate_gpu(Activation TYPE);
GradientGpu load_gradient_gpu(Activation TYPE);

void activate_list_gpu(float *origin, int num, ActivateGpu FUNC);
void gradient_list_gpu(float *origin, int num, GradientGpu FUNC);

#ifdef  __cplusplus
}
#endif

#endif