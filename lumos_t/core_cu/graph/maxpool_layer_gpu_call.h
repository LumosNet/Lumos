#ifndef MAXPOOL_LAYER_GPU_CALL_H
#define MAXPOOL_LAYER_GPU_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include "layer.h"
#include "maxpool_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_forward_maxpool_layer_gpu(void **params, void **ret);
void call_backward_maxpool_layer_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
