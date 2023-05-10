#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "cpu.h"
#include "bias.h"
#include "normalize.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_normalization_layer();

void init_normalization_layer(Layer *l, int w, int h, int c);

void forward_normalization_layer(Layer l, int num);
void backward_normalization_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif
