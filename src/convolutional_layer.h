#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "lumos.h"
#include "image.h"
#include "active.h"
#include "convolution.h"

#ifdef __cplusplus
extern "C"{
#endif

void forward_convolutional_layer(Layer *l, Network *net);
void backward_convolutional_layer(Layer *l, Network *net);

#ifdef __cplusplus
}
#endif

#endif