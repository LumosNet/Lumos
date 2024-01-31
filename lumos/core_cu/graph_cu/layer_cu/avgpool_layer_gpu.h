#ifndef AVGPOOL_LAYER_GPU_H
#define AVGPOOL_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"
#include "pooling_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_avgpool_layer_gpu(Layer *l, int w, int h, int c, int subdivision);
void forward_avgpool_layer_gpu(Layer l, int num);
void backward_avgpool_layer_gpu(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif
#endif