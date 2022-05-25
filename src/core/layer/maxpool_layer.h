#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include <string.h>

#include "im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_maxpool_layer(CFGParams *p, int h, int w, int c);

void forward_maxpool_layer(Layer l, float *workspace);
void backward_maxpool_layer(Layer l, float *n_delta, float *workspace);

#ifdef __cplusplus
}
#endif

#endif