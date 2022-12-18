#ifndef GEMM_CALL_H
#define GEMM_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_gemm(void **params, void **ret);

void call_gemm_nn(void **params, void **ret);
void call_gemm_tn(void **params, void **ret);
void call_gemm_nt(void **params, void **ret);
void call_gemm_tt(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
