#ifndef IM2COL_LAYER_H
#define IM2COL_LAYER_H

#include "lumos.h"
#include "parser.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_im2col_layer(LayerParams *p, int batch, int h, int w, int c);

void forward_im2col_layer(Layer l, Network net);
void backward_im2col_layer(Layer l, Network net);

#ifdef __cplusplus
}
#endif

#endif