#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "image.h"
#include "active.h"
#include "bias.h"
#include "gemm.h"
#include "cpu.h"

#include "convolutional_layer_gpu.h"

#ifdef __cplusplus
extern "C"{
#endif

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, char *active);
void init_convolutional_layer(Layer *l, int w, int h, int c, int subdivision);
void weightinit_convolutional_layer(Layer l, FILE *fp);

void forward_convolutional_layer(Layer l, int num);
void backward_convolutional_layer(Layer l, float rate, int num, float *n_delta);
void update_convolutional_layer(Layer l, float rate, int num, float *n_delta);
void update_convolutional_layer_weights(Layer l);

void save_convolutional_layer_weights(Layer l, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif