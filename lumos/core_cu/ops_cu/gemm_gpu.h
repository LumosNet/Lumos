#ifndef GEMM_GPU_H
#define GEMM_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GEMM_BLOCK 16

void gemm_gpu(int TA, int TB, int AM, int AN, int BM, int BN, float ALPHA, 
        float *A, float *B, float *C);

void gemm_nn_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_tn_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_nt_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_tt_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);

#ifdef __cplusplus
}
#endif
#endif
