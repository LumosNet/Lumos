#ifndef CONNECT_LAYER_H
#define CONNECT_LAYER_H

#include <time.h>

#include "bias.h"
#include "active.h"
#include "gemm.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_connect_layer(CFGParams *p, int h, int w, int c);

void forward_connect_layer(Layer l, float *workspace);
void backward_connect_layer(Layer l, float *n_delta, float *workspace);

void update_connect_layer(Layer l, float rate, float *n_delta, float *workspace);

void save_connect_weights(Layer l, FILE *file);
void load_connect_weights(Layer l, FILE *file);

#ifdef __cplusplus
}
#endif

#endif