#ifndef MSE_LAYER_H
#define MSE_LAYER_H

#include "lumos.h"
#include "parser.h"
#include "tensor.h"
#include "array.h"
#include "vector.h"
#include "gemm.h"
#include "umath.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_mse_layer(LayerParams *p, int batch, int h, int w, int c);

void forward_mse_layer(Layer l, Network net);
void backward_mse_layer(Layer l, Network net);

#ifdef __cplusplus
}
#endif

#endif