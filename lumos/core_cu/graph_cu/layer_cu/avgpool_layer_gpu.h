#ifndef AVGPOOL_LAYER_GPU_H
#define AVGPOOL_LAYER_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"
#include "pooling_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_avgpool_layer_gpu(Layer *l, int w, int h, int c);
void forward_avgpool_layer_gpu(Layer l, int num);
void backward_avgpool_layer_gpu(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif
#endif