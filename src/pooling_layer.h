#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <string.h>

#include "lumos.h"
#include "parser.h"
#include "tensor.h"
#include "avgpool_layer.h"
#include "maxpool_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

PoolingType load_pooling_type(char *pool);

Layer make_pooling_layer(LayerParams *p, int batch, int h, int w, int c);

#ifdef __cplusplus
}
#endif

#endif