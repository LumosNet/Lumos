#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <math.h>
#include <float.h>

#include "layer.h"
#include "cfg_f.h"
#include "gemm.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_softmax_layer(CFGParams *p, int h, int w, int c);

void forward_softmax_layer(Layer l, float *workspace);
void backward_softmax_layer(Layer l, float *n_delta, float *workspace);

void softmax_cpu(float *input, int n, int batch, int batch_offset, float *output);
void softmax(float *input, int n, float *output);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);

#ifdef __cplusplus
}
#endif

#endif