#ifndef BIAS_GPU_H
#define BIAS_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

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
