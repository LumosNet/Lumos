#ifndef CONNECT_LAYER_GPU_H
#define CONNECT_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "binary_f.h"
#include "gpu.h"
#include "layer.h"
#include "random.h"
#include "active_gpu.h"
#include "bias_gpu.h"
#include "cpu_gpu.h"
#include "gemm_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_connect_layer_gpu(int output, int bias, int normalize, char *active);
void init_connect_layer_gpu(Layer *l, int w, int h, int c);
void release_connect_layer_gpu(Layer *l);
void weightinit_connect_layer_gpu(Layer *l, WeightInitType type);

void forward_connect_layer_gpu(Layer l, int num);
void backward_connect_layer_gpu(Layer l, float rate, int num, float *n_delta);
void update_connect_layer_gpu(Layer l, float rate, int num, float *n_delta);
void update_connect_layer_weights_gpu(Layer *l);

void load_connect_layer_weights_gpu(Layer *l, FILE *p);
void save_connect_layer_weights_gpu(Layer *l, FILE *p);

#ifdef __cplusplus
}
#endif
#endif
