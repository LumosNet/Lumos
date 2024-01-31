#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layer.h"
#include "im2col.h"
#include "cpu.h"
#include "pooling.h"

#include "maxpool_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_maxpool_layer(int ksize, int stride, int pad);

void init_maxpool_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_maxpool_layer(Layer l, int num);
void backward_maxpool_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif