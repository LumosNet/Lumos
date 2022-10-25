#include "gemm.h"

void gemm(int TA, int TB, int AM, int AN, int BM, int BN, float ALPHA,
          float *A, float *B, float *C)
{
    if (!TA && !TB)
    {
        fill_cpu(C, AM * BN, 0, 1);
        gemm_nn(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else if (TA && !TB)
    {
        fill_cpu(C, AN * BN, 0, 1);
        gemm_tn(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else if (!TA && TB)
    {
        fill_cpu(C, AM * BM, 0, 1);
        gemm_nt(AM, AN, BM, BN, ALPHA, A, B, C);
    }
    else
    {
        fill_cpu(C, AN * BM, 0, 1);
        gemm_tt(AM, AN, BM, BN, ALPHA, A, B, C);
    }
}

void gemm_nn(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    #pragma omp parallel for
    for (int i = 0; i < AM; ++i)
    {
        for (int j = 0; j < AN; ++j)
        {
            register float temp = ALPHA * A[i * AN + j];
            for (int k = 0; k < BN; ++k)
            {
                C[i * BN + k] += temp * B[j * BN + k];
            }
        }
    }
}

void gemm_tn(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    #pragma omp parallel for
    for (int i = 0; i < AN; ++i)
    {
        for (int j = 0; j < AM; ++j)
        {
            register float temp = ALPHA * A[j * AN + i];
            for (int k = 0; k < BN; ++k)
            {
                C[i * BN + k] += temp * B[j * BN + k];
            }
        }
    }
}

void gemm_nt(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    #pragma omp parallel for
    for (int i = 0; i < AM; ++i)
    {
        for (int j = 0; j < AN; ++j)
        {
            register float temp = ALPHA * A[i * AN + j];
            for (int k = 0; k < BM; ++k)
            {
                C[i * BM + k] += temp * B[k * BN + j];
            }
        }
    }
}

void gemm_tt(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C)
{
    #pragma omp parallel for
    for (int i = 0; i < AN; ++i)
    {
        for (int j = 0; j < AM; ++j)
        {
            register float temp = ALPHA * A[j * AN + i];
            for (int k = 0; k < BM; ++k)
            {
                C[i * BM + k] += temp * B[k * BN + j];
            }
        }
    }
}
