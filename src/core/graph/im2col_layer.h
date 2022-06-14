#ifndef IM2COL_LAYER_H
#define IM2COL_LAYER_H

#include "layer.h"
#include "cfg_f.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_im2col_layer(int flag);
Layer make_im2col_layer_by_cfg(CFGParams *p);

void init_im2col_layer(Layer l, int w, int h, int c);
void restore_im2col_layer(Layer l);

void forward_im2col_layer(Layer l);
void backward_im2col_layer(Layer l, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif