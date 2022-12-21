#include "gemm_call.h"

void call_gemm(void **params, void **ret)
{
    int *TA = (int*)params[0];
    int *TB = (int*)params[1];
    int *AM = (int*)params[2];
    int *AN = (int*)params[3];
    int *BM = (int*)params[4];
    int *BN = (int*)params[5];
    float *ALPHA = (float*)params[6];
    float *A = (float*)params[7];
    float *B = (float*)params[8];
    float *C = (float*)params[9];
    gemm(TA[0], TB[0], AM[0], AN[0], BM[0], BN[0], ALPHA[0], \
         A, B, C);
    ret[0] = (void*)C;
}

void call_gemm_nn(void **params, void **ret)
{
    int *AM = (int*)params[0];
    int *AN = (int*)params[1];
    int *BM = (int*)params[2];
    int *BN = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *A = (float*)params[5];
    float *B = (float*)params[6];
    float *C = (float*)params[7];
    gemm_nn(AM[0], AN[0], BM[0], BN[0], ALPHA[0], A, B, C);
    ret[0] = (void*)C;
}

void call_gemm_tn(void **params, void **ret)
{
    int *AM = (int*)params[0];
    int *AN = (int*)params[1];
    int *BM = (int*)params[2];
    int *BN = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *A = (float*)params[5];
    float *B = (float*)params[6];
    float *C = (float*)params[7];
    gemm_tn(AM[0], AN[0], BM[0], BN[0], ALPHA[0], A, B, C);
    ret[0] = (void*)C;
}

void call_gemm_nt(void **params, void **ret)
{
    int *AM = (int*)params[0];
    int *AN = (int*)params[1];
    int *BM = (int*)params[2];
    int *BN = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *A = (float*)params[5];
    float *B = (float*)params[6];
    float *C = (float*)params[7];
    gemm_nt(AM[0], AN[0], BM[0], BN[0], ALPHA[0], A, B, C);
    ret[0] = (void*)C;
}

void call_gemm_tt(void **params, void **ret)
{
    int *AM = (int*)params[0];
    int *AN = (int*)params[1];
    int *BM = (int*)params[2];
    int *BN = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *A = (float*)params[5];
    float *B = (float*)params[6];
    float *C = (float*)params[7];
    gemm_tt(AM[0], AN[0], BM[0], BN[0], ALPHA[0], A, B, C);
    ret[0] = (void*)C;
}
