#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include <string.h>

#include "layer.h"
#include "cfg_f.h"
#include "im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_maxpool_layer(CFGParams *p);

void forward_maxpool_layer(Layer l);
void backward_maxpool_layer(Layer l, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif