#ifndef CONVOLUTIONAL_LAYER_GPU_H
#define CONVOLUTIONAL_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "random.h"
#include "cpu.h"
#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"
#include "active_gpu.h"
#include "gemm_gpu.h"
#include "im2col_gpu.h"
#include "bias_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_convolutional_layer_gpu(Layer *l, int w, int h, int c, int subdivision);
void weightinit_convolutional_layer_gpu(Layer l, FILE *fp);
void forward_convolutional_layer_gpu(Layer l, int num);
void backward_convolutional_layer_gpu(Layer l, float rate, int num, float *n_delta);
void update_convolutional_layer_gpu(Layer l, float rate, int num, float *n_delta);
void update_convolutional_layer_weights_gpu(Layer l);

void save_convolutional_layer_weights_gpu(Layer l, FILE *fp);

#ifdef __cplusplus
}
#endif
#endif
