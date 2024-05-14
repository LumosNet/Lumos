#ifndef CONNECT_LAYER_H
#define CONNECT_LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "bias.h"
#include "active.h"
#include "gemm.h"
#include "cpu.h"

#include "connect_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_connect_layer(int output, int bias, char *active);
void init_connect_layer(Layer *l, int w, int h, int c, int subdivision);
void weightinit_connect_layer(Layer l, FILE *fp);

void forward_connect_layer(Layer l, int num);
void backward_connect_layer(Layer l, float rate, int num, float *n_delta);
void update_connect_layer(Layer l, float rate, int num, float *n_delta);
void update_connect_layer_weights(Layer l);

void save_connect_layer_weights(Layer l, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif