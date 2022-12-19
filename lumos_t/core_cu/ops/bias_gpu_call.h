#ifndef BIAS_GPU_CALL_H
#define BIAS_GPU_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bias_gpu.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GPU
void call_add_bias_gpu(void **params, void **ret);
#endif

#ifdef __cplusplus
}
#endif

#endif
