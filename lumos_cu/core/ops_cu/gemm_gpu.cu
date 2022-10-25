#include "gemm_gpu.h"

__global__ void gemm_nn_kernel(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    float temp = ALPHA * A[i * AN + j];
    C[i * BN + k] += temp * B[j * BN + k];
}

void gemm_gpu(int TA, int TB, int AM, int AN, int BM, int BN, float ALPHA, 
        float *A, float *B, float *C)
{
    if (!TA && !TB)
    {
        fill_gpu(C, AM * BN, 0, 1);
        gemm_nn_gpu(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else if (TA && !TB)
    {
        fill_gpu(C, AN * BN, 0, 1);
        gemm_tn_gpu(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else if (!TA && TB)
    {
        fill_gpu(C, AM * BM, 0, 1);
        gemm_nt_gpu(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else
    {
        fill_gpu(C, AN * BM, 0, 1);
        gemm_tt_gpu(AM, AN, BM, BN, ALPHA, A, B, C);
    }
}

void gemm_nn_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    dim3 dimGrid(AM, AN, BN);
    dim3 dimBlock(BLOCK, 1, 1);
    gemm_nn_kernel(AM, AN, BM, BN, ALPHA, A, B, C);
}

void gemm_tn_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{

}

void gemm_nt_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{

}

void gemm_tt_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{

}

