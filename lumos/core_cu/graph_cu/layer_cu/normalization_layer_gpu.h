#ifndef NORMALIZATION_LAYER_GPU_H
#define NORMALIZATION_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdlib.h>
#include <stdio.h>

#include "gpu.h"
#include "cpu_gpu.h"
#include "layer.h"
#include "normalize_gpu.h"
#include "bias_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_normalization_layer_gpu(Layer *l, int subdivision);
void weightinit_normalization_layer_gpu(Layer l, FILE *fp);

void forward_normalization_layer_gpu(Layer l, int num);
void backward_normalization_layer_gpu(Layer l, float rate, int num, float *n_delta);
void update_normalization_layer_gpu(Layer l, float rate, int num, float *n_delta);
void update_normalization_layer_weights_gpu(Layer l);

void save_normalization_layer_weights_gpu(Layer l, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif