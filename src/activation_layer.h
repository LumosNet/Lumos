#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "lumos.h"
#include "parser.h"
#include "tensor.h"
#include "loss.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_activation_layer(LayerParams *p, int batch, int h, int w, int c);

#ifdef __cplusplus
}
#endif

#endif