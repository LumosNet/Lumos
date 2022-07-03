#ifndef IM2COL_LAYER_H
#define IM2COL_LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "cfg_f.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_im2col_layer(int flag);
Layer *make_im2col_layer_by_cfg(CFGParams *p);

void init_im2col_layer(Layer *l, int w, int h, int c);

void forward_im2col_layer(Layer l, int num);
void backward_im2col_layer(Layer l, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif