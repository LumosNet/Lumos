#ifndef CPU_GPU_CALL_H
#define CPU_GPU_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include "cpu_gpu.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_fill_gpu(void **params, void **ret);
void call_multy_gpu(void **params, void **ret);
void call_add_gpu(void **params, void **ret);

void call_min_gpu(void **params, void **ret);
void call_max_gpu(void **params, void **ret);
void call_sum_gpu(void **params, void **ret);
void call_mean_gpu(void **params, void **ret);

void call_matrix_add_gpu(void **params, void **ret);
void call_matrix_subtract_gpu(void **params, void **ret);
void call_matrix_multiply_gpu(void **params, void **ret);
void call_matrix_divide_gpu(void **params, void **ret);

void call_saxpy_gpu(void **params, void **ret);
void call_sum_channel_gpu(void **params, void **ret);

// void call_exp_list_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
