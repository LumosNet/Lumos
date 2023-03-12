#ifndef MSE_LAYER_CALL_H
#define MSE_LAYER_CALL_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "mse_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_forward_mse_layer(void **params, void **ret);
void call_backward_mse_layer(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
