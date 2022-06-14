#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "layer.h"
#include "cfg_f.h"
#include "im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_avgpool_layer(int ksize);
Layer *make_avgpool_layer_by_cfg(CFGParams *p);

void init_avgpool_layer(Layer *l, int w, int h, int c);
void restore_avgpool_layer(Layer *l);

void forward_avgpool_layer(Layer l);
void backward_avgpool_layer(Layer l, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif