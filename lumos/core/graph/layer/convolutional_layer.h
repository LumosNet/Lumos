#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

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

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, int normalization, char *active);
void init_convolutional_layer(Layer *l, int w, int h, int c);

void forward_convolutional_layer(Layer l, int num);
void backward_convolutional_layer(Layer l, float rate, int num);
void update_convolutional_layer(Layer l, float rate, int num);

#ifdef __cplusplus
}
#endif

#endif