#ifndef IM2COL_LAYER_CALL_H
#define IM2COL_LAYER_CALL_H

#include "layer.h"
#include "im2col_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_forward_im2col_layer(void **params, void **ret);
void call_backward_im2col_layer(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
