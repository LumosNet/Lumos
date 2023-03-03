#include "gemm_gpu.h"

__global__ void gemm_nn_kernel(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= AM || j >= BN) return;
    float res = 0;
    for (int k = 0; k < AN; ++k){
        res += ALPHA * A[i * AN + k] * B[k * BN + j];
    }
    C[i * BN + j] = res;
}

__global__ void gemm_tn_kernel(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= AN || j >= BN) return;
    float res = 0;
    for (int k = 0; k < AM; ++k){
        res += ALPHA * A[k * AN + i] * B[k * BN + j];
    }
    C[i * BN + j] = res;
}

__global__ void gemm_nt_kernel(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= AM || j >= BM) return;
    float res = 0;
    for (int k = 0; k < AN; ++k){
        res += ALPHA * A[i * AN + k] * B[j * BN + k];
    }
    C[i * BM + j] = res;
}

__global__ void gemm_tt_kernel(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= AN || j >= BM) return;
    float res = 0;
    for (int k = 0; k < AM; ++k){
        res += ALPHA * A[k * AN + i] * B[j * BN + k];
    }
    C[i * BM + j] = res;
}

void gemm_gpu(int TA, int TB, int AM, int AN, int BM, int BN, float ALPHA, 
        float *A, float *B, float *C)
{
    if (!TA && !TB)
    {
        gemm_nn_gpu(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else if (TA && !TB)
    {
        gemm_tn_gpu(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else if (!TA && TB)
    {
        gemm_nt_gpu(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else
    {
        gemm_tt_gpu(AM, AN, BM, BN, ALPHA, A, B, C);
    }
}

void gemm_nn_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    dim3 dimGrid((AM+GEMM_BLOCK-1)/GEMM_BLOCK, (BN+GEMM_BLOCK-1)/GEMM_BLOCK);
    dim3 dimBlock(GEMM_BLOCK, GEMM_BLOCK);
    gemm_nn_kernel<<<dimGrid, dimBlock>>>(AM, AN, BM, BN, ALPHA, A, B, C);
}

void gemm_tn_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    dim3 dimGrid((AN+GEMM_BLOCK-1)/GEMM_BLOCK, (BN+GEMM_BLOCK-1)/GEMM_BLOCK);
    dim3 dimBlock(GEMM_BLOCK, GEMM_BLOCK);
    gemm_tn_kernel<<<dimGrid, dimBlock>>>(AM, AN, BM, BN, ALPHA, A, B, C);
}

void gemm_nt_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    dim3 dimGrid((AM+GEMM_BLOCK-1)/GEMM_BLOCK, (BM+GEMM_BLOCK-1)/GEMM_BLOCK);
    dim3 dimBlock(GEMM_BLOCK, GEMM_BLOCK);
    gemm_nt_kernel<<<dimGrid, dimBlock>>>(AM, AN, BM, BN, ALPHA, A, B, C);
}

void gemm_tt_gpu(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    dim3 dimGrid((AN+GEMM_BLOCK-1)/GEMM_BLOCK, (BM+GEMM_BLOCK-1)/GEMM_BLOCK);
    dim3 dimBlock(GEMM_BLOCK, GEMM_BLOCK);
    gemm_tt_kernel<<<dimGrid, dimBlock>>>(AM, AN, BM, BN, ALPHA, A, B, C);
}

