#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <string.h>

#include "lumos.h"
#include "pooling.h"

#ifdef __cplusplus
extern "C" {
#endif

PoolingType load_pooling_type(char *pool);

void forward_pooling_layer(Layer *l, Network *net);
void backward_pooling_layer(Layer *l, Network *net);

#ifdef __cplusplus
}
#endif

#endif