#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "binary_f.h"
#include "layer.h"
#include "image.h"
#include "active.h"
#include "bias.h"
#include "gemm.h"
#include "cpu.h"
#include "random.h"
#include "normalization_layer.h"

#include "convolutional_layer_gpu.h"

#ifdef __cplusplus
extern "C"{
#endif

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, int normalization, char *active);
void init_convolutional_layer(Layer *l, int w, int h, int c);
void release_convolutional_layer(Layer *l);
void weightinit_convolutional_layer(Layer *l, WeightInitType type);

void forward_convolutional_layer(Layer l, int num);
void backward_convolutional_layer(Layer l, float rate, int num, float *n_delta);
void update_convolutional_layer(Layer l, float rate, int num, float *n_delta);
void update_convolutional_layer_weights(Layer *l);

void load_convolutional_layer_weights(Layer *l, FILE *p);
void save_convolutional_layer_weights(Layer *l, FILE *p);

#ifdef __cplusplus
}
#endif

#endif