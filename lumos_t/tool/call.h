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

#ifdef GPU
#include "bias_gpu_call.h"
#include "cpu_gpu_call.h"
#include "gemm_gpu_call.h"
#include "im2col_gpu_call.h"
#include "pooling_gpu_call.h"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

void call_ops(char *interface, void **params, void **ret);

#ifdef GPU
void call_cu_ops(char *interface, void **params, void **ret);
#endif

#ifdef __cplusplus
}
#endif

#endif
