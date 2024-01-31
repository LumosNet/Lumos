#ifndef IM2COL_GPU_CALL_H
#define IM2COL_GPU_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include "im2col_gpu.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_im2col_gpu(void **params, void **ret);
void call_col2im_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
