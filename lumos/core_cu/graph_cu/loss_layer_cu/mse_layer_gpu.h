#ifndef MSE_LAYER_GPU_H
#define MSE_LAYER_GPU_H

#ifdef GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include "cpu.h"
#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"
#include "gemm_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU
void forward_mse_layer_gpu(Layer l, int num);
void backward_mse_layer_gpu(Layer l, float rate, int num, float *n_delta);
#endif

#ifdef __cplusplus
}
#endif
#endif
