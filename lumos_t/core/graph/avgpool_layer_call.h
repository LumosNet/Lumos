#ifndef AVGPOOL_LAYER_CALL_H
#define AVGPOOL_LAYER_CALL_H

#include "layer.h"
#include "avgpool_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_forward_avgpool_layer(void **params, void **ret);
void call_backward_avgpool_layer(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
