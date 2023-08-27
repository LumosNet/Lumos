#ifndef NORMALIZATION_LAYER_GPU_H
#define NORMALIZATION_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "layer.h"
#include "normalize_gpu.h"
#include "cpu_gpu.h"
#include "bias_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_normalization_layer_gpu(Layer l, int num);
void backward_normalization_layer_gpu(Layer l, float rate, int num);
void update_normalization_layer_gpu(Layer l, float rate, int num);

#ifdef __cplusplus
}
#endif
#endif
