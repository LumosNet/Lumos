#ifndef IM2COL_LAYER_H
#define IM2COL_LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "cpu.h"

#include "im2col_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_im2col_layer();

void init_im2col_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_im2col_layer(Layer l, int num);
void backward_im2col_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif