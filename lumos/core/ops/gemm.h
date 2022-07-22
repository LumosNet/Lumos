#ifndef GEMM_H
#define GEMM_H

#include "cpu.h"

#ifdef __cplusplus
extern "C"{
#endif

void gemm(int TA, int TB, int AM, int AN, int BM, int BN, float ALPHA, 
        float *A, float *B, float *C);

void gemm_nn(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_tn(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_nt(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_tt(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);

#ifdef __cplusplus
}
#endif

#endif