#ifndef CALL_H
#define CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bias_call.h"
#include "cpu_call.h"
#include "gemm_call.h"
#include "im2col_call.h"
#include "image_call.h"
#include "pooling_call.h"

#include "avgpool_layer_call.h"
#include "batchnorm_layer_call.h"
#include "connect_layer_call.h"
#include "convolutional_layer_call.h"
#include "im2col_layer_call.h"
#include "maxpool_layer_call.h"
#include "mse_layer_call.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_ops(char *interface, void **params, void **ret);
void call_graph(char *interface, void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
