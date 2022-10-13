#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layer.h"
#include "cfg_f.h"
#include "im2col.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_maxpool_layer(int ksize);
Layer *make_maxpool_layer_by_cfg(CFGParams *p);

void init_maxpool_layer(Layer *l, int w, int h, int c);

void forward_maxpool_layer(Layer l, int num);
void backward_maxpool_layer(Layer l, float rate, int num, float *n_delta);

int get_f_calculate_t(Layer l);

#ifdef __cplusplus
}
#endif

#endif