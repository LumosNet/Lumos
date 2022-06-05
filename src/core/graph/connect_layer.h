#ifndef CONNECT_LAYER_H
#define CONNECT_LAYER_H

#include "layer.h"
#include "cfg_f.h"
#include "bias.h"
#include "active.h"
#include "gemm.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_connect_layer(CFGParams *p);

void init_connect_layer(Layer l, int w, int h, int c);
void restore_connect_layer(Layer l);

void forward_connect_layer(Layer l);
void backward_connect_layer(Layer l, float *n_delta);

void update_connect_layer(Layer l, float rate, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif