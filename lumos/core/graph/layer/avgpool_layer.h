#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "layer.h"
#include "im2col.h"
#include "cpu.h"
#include "pooling.h"

#include "avgpool_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_avgpool_layer(int ksize, int stride, int pad);

void init_avgpool_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_avgpool_layer(Layer l, int num);
void backward_avgpool_layer(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif