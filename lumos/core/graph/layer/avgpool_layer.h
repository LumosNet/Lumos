#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "layer.h"
#include "cfg_f.h"
#include "im2col.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_avgpool_layer(int ksize);
Layer *make_avgpool_layer_by_cfg(CFGParams *p);

void init_avgpool_layer(Layer *l, int w, int h, int c);

void forward_avgpool_layer(Layer l, int num);
void backward_avgpool_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif