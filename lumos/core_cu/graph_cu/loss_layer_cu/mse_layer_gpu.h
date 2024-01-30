#ifndef MSE_LAYER_GPU_H
#define MSE_LAYER_GPU_H

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

void init_mse_layer_gpu(Layer *l, int w, int h, int c);
void forward_mse_layer_gpu(Layer l, int num);
void backward_mse_layer_gpu(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif
#endif
