#ifndef IM2COL_LAYER_GPU_H
#define IM2COL_LAYER_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_im2col_layer_gpu(Layer *l, int w, int h, int c);
void forward_im2col_layer_gpu(Layer l, int num);
void backward_im2col_layer_gpu(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif
#endif
