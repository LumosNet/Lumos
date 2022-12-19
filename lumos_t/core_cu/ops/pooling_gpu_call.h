#ifndef POOLING_GPU_CALL_H
#define POOLING_GPU_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pooling_gpu.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GPU
void call_avgpool_gpu(void **params, void **ret);
void call_maxpool_gpu(void **params, void **ret);

void call_avgpool_gradient_gpu(void **params, void **ret);
void call_maxpool_gradient_gpu(void **params, void **ret);
#endif

#ifdef __cplusplus
}
#endif

#endif
