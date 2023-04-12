#ifndef ACTIVE_GPU_H
#define ACTIVE_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

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

void activate_list_gpu(float *origin, int num, Activation TYPE);
void gradient_list_gpu(float *origin, int num, Activation TYPE);

#ifdef  __cplusplus
}
#endif

#endif