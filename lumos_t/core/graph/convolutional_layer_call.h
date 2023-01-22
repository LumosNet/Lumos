#ifndef CONVOLUTIONAL_LAYER_CALL_H
#define CONVOLUTIONAL_LAYER_CALL_H

#include "layer.h"
#include "convolutional_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_forward_convolutional_layer(void **params, void **ret);
void call_backward_convolutional_layer(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
