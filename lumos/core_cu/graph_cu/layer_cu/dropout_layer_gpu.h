#ifndef DROPOUT_LAYER_GPU_H
#define DROPOUT_LAYER_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"
#include "active_gpu.h"
#include "gemm_gpu.h"
#include "im2col_gpu.h"
#include "bias_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_dropout_layer_gpu(Layer l, int num);
void backward_dropout_layer_gpu(Layer l, float rate, int num, float *n_delta);

void dropout_gpu(Layer l, int num);
void dropout_gradient_gpu(Layer l, int num, float *n_delta);

#ifdef __cplusplus
}
#endif
#endif
