#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <math.h>
#include <float.h>

#include "lumos.h"
#include "parser.h"
#include "umath.h"
#include "gemm.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_softmax_layer(Layer l, Network net);
void backward_softmax_layer(Layer l, Network net);

Layer make_softmax_layer(LayerParams *p, int batch, int h, int w, int c);

void softmax_cpu(float *input, int n, int batch, int batch_offset, float *output);
void softmax(float *input, int n, float *output);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);

#ifdef __cplusplus
}
#endif

#endif