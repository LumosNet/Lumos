#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "random.h"

#include "dropout_layer_gpu.h"

#ifdef __cplusplus
extern "C"{
#endif

Layer *make_dropout_layer(float probability);

void init_dropout_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_dropout_layer(Layer l, int num);
void backward_dropout_layer(Layer l, float rate, int num, float *n_delta);

void free_dropout_layer(Layer l);

#ifdef __cplusplus
}
#endif

#endif