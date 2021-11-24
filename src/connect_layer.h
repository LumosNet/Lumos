#ifndef CONNECT_LAYER_H
#define CONNECT_LAYER_H

#include "lumos.h"
#include "parser.h"
#include "array.h"
#include "bias.h"
#include "active.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_connect_layer(Layer *l, Network *net);
void backward_connect_layer(Layer *l, Network *net);

Layer *make_connect_layer(LayerParams *p, int h, int w, int c);

#ifdef __cplusplus
}
#endif

#endif