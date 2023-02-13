#ifndef MSE_LAYER_H
#define MSE_LAYER_H

#include "layer.h"
#include "cfg_f.h"
#include "cpu.h"
#include "gemm.h"

#ifdef GPU
#include "mse_layer_gpu.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_mse_layer(int group);

void init_mse_layer(Layer *l, int w, int h, int c);

void forward_mse_layer(Layer l, int num);
void backward_mse_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif