#ifndef CONNECT_LAYER_H
#define CONNECT_LAYER_H

#include "lumos.h"
#include "parser.h"
#include "array.h"
#include "bias.h"
#include "active.h"
#include "gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_connect_layer(Layer l, Network net);
void backward_connect_layer(Layer l, Network net);

Layer make_connect_layer(LayerParams *p, int batch, int h, int w, int c);

void update_connect_layer(Layer l, Network net);

// void save_connect_weights(Layer *l, FILE *file);
// void load_connect_weights(Layer *l, FILE *file);

#ifdef __cplusplus
}
#endif

#endif