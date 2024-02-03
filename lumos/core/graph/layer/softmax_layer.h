#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "softmax.h"
#include "cpu.h"

#include "softmax_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_softmax_layer(int group);

void init_softmax_layer(Layer *l, int w, int h, int c, int subdivision);

void forward_softmax_layer(Layer l, int num);
void backward_softmax_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif