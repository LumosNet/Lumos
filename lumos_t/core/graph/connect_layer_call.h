#ifndef CONNECT_LAYER_CALL_H
#define CONNECT_LAYER_CALL_H

#include "layer.h"
#include "connect_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_forward_connect_layer(void **params, void **ret);
void call_backward_connect_layer(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
