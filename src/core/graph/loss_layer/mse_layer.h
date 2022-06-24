#ifndef MSE_LAYER_H
#define MSE_LAYER_H

#include "layer.h"
#include "cfg_f.h"
#include "cpu.h"
#include "gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_mse_layer(int group);
Layer *make_mse_layer_by_cfg(CFGParams *p);

void init_mse_layer(Layer *l, int w, int h, int c);
void restore_mse_layer(Layer *l);

void forward_mse_layer(Layer l, int num);
void backward_mse_layer(Layer l, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif