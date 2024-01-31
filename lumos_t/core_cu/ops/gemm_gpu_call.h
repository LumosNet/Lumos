#ifndef GEMM_GPU_CALL_H
#define GEMM_GPU_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include "gemm_gpu.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_gemm_gpu(void **params, void **ret);

void call_gemm_nn_gpu(void **params, void **ret);
void call_gemm_tn_gpu(void **params, void **ret);
void call_gemm_nt_gpu(void **params, void **ret);
void call_gemm_tt_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
