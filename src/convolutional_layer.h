#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "lumos.h"
#include "parser.h"
#include "image.h"
#include "active.h"
#include "convolution.h"
#include "bias.h"

#ifdef __cplusplus
extern "C"{
#endif

void forward_convolutional_layer(Layer *l, Network *net);
void backward_convolutional_layer(Layer *l, Network *net);

Layer *make_convolutional_layer(Network *net, LayerParams *p, int h, int w, int c);

void save_convolutional_weights(Layer *l, FILE *file);
void load_convolutional_weights(Layer *l, FILE *file);

#ifdef __cplusplus
}
#endif

#endif