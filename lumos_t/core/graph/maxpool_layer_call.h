#ifndef MAXPOOL_LAYER_CALL_H
#define MAXPOOL_LAYER_CALL_H

#include "layer.h"
#include "maxpool_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_forward_maxpool_layer(void **params, void **ret);
void call_backward_maxpool_layer(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
