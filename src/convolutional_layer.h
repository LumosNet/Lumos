#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "lumos.h"
#include "image.h"
#include "active.h"
#include "convolution.h"
#include "bias.h"

#ifdef __cplusplus
extern "C"{
#endif

void convolutional_layer(Layer *l, Network *net);
void backward_convolutional_layer(Layer *l, Network *net);

#ifdef __cplusplus
}
#endif

#endif