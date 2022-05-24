#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include <string.h>

#include "im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_maxpool_layer(LayerParams *p, int batch, int h, int w, int c);

void forward_maxpool_layer(Layer l, Network net);
void backward_maxpool_layer(Layer l, Network net);

#ifdef __cplusplus
}
#endif

#endif