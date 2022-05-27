#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include <string.h>

#include "layer.h"
#include "cfg_f.h"
#include "im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_avgpool_layer(CFGParams *p, int h, int w, int c);

void forward_avgpool_layer(Layer l, float *workspace);
void backward_avgpool_layer(Layer l, float *n_delta, float *workspace);

#ifdef __cplusplus
}
#endif

#endif