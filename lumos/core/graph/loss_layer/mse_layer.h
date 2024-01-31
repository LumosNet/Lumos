#ifndef MSE_LAYER_H
#define MSE_LAYER_H

#include "layer.h"
#include "cpu.h"
#include "gemm.h"

#include "mse_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_mse_layer(int group);

void init_mse_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_mse_layer(Layer l, int num);
void backward_mse_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif