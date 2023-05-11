#ifndef CONNECT_LAYER_H
#define CONNECT_LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "bias.h"
#include "active.h"
#include "gemm.h"
#include "cpu.h"
#include "normalization_layer.h"

#include "connect_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif


Layer *make_connect_layer(int output, int bias, int normalize, char *active);
void init_connect_layer(Layer *l, int w, int h, int c);

void forward_connect_layer(Layer l, int num);
void backward_connect_layer(Layer l, float rate, int num, float *n_delta);

void update_connect_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif