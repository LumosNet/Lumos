#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <math.h>

#include "lumos.h"
#include "parser.h"
#include "tensor.h"
#include "array.h"
#include "umath.h"
#include "gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_softmax_layer(Layer l, Network net);
void backward_softmax_layer(Layer l, Network net);

Layer make_softmax_layer(LayerParams *p, int batch, int h, int w, int c);

#ifdef __cplusplus
}
#endif

#endif