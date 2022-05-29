#ifndef MSE_LAYER_H
#define MSE_LAYER_H

#include "layer.h"
#include "cfg_f.h"
#include "gemm.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_mse_layer(CFGParams *p);

void forward_mse_layer(Layer l);
void backward_mse_layer(Layer l, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif