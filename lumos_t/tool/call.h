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

#include "bias_gpu_call.h"
#include "cpu_gpu_call.h"
#include "gemm_gpu_call.h"
#include "im2col_gpu_call.h"
#include "pooling_gpu_call.h"

#include "avgpool_layer_gpu_call.h"
#include "batchnorm_layer_gpu_call.h"
#include "connect_layer_gpu_call.h"
#include "convolutional_layer_gpu_call.h"
#include "im2col_layer_gpu_call.h"
#include "maxpool_layer_gpu_call.h"

#include "dropout_rand_call.h"

#include "utest.h"

#ifdef  __cplusplus
extern "C" {
#endif

int call(char *interface, void **params, void **ret);
int call_cu(char *interface, void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
