#ifndef NORMALIZATION_LAYER_GPU_H
#define NORMALIZATION_LAYER_GPU_H

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
void backward_normalization_layer_gpu(Layer l, float rate, int num, float *n_delta);
void update_normalization_layer_gpu(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif
#endif
