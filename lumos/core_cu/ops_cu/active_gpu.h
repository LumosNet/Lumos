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

Activate load_activate_gpu(char *activate);
Gradient load_gradient_gpu(char *activate);

void activate_list_gpu(float *origin, int num, Activate func);
void gradient_list_gpu(float *origin, int num, Gradient func);

#ifdef  __cplusplus
}
#endif

#endif