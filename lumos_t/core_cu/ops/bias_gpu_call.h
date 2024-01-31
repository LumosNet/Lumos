#ifndef BIAS_GPU_CALL_H
#define BIAS_GPU_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include "bias_gpu.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_add_bias_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
