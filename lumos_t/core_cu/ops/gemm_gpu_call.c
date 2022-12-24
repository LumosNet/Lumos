#include "gemm_gpu_call.h"

void call_gemm_gpu(void **params, void **ret)
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
    float *A_g = NULL;
    float *B_g = NULL;
    float *C_g = NULL;
    int c_size = 0;
    cudaMalloc((void**)&A_g, AM[0]*AN[0]*sizeof(float));
    cudaMalloc((void**)&B_g, BM[0]*BN[0]*sizeof(float));
    if (TA[0] == 0 && TB[0] == 0){
        c_size = AM[0]*BN[0];
    } else if (TA[0] == 0 && TB[0] == 1){
        c_size = AM[0]*BM[0];
    } else if (TA[0] == 1 && TB[0] == 0){
        c_size = AN[0]*BN[0];
    } else if (TA[0] == 1 && TB[0] == 1){
        c_size = AN[0]*BM[0];
    }
    cudaMalloc((void**)&C_g, c_size*sizeof(float));
    cudaMemcpy(A_g, A, AM[0]*AN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_g, B, BM[0]*BN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_g, C, c_size*sizeof(float), cudaMemcpyHostToDevice);
    gemm_gpu(TA[0], TB[0], AM[0], AN[0], BM[0], BN[0], ALPHA[0], A_g, B_g, C_g);
    cudaMemcpy(C, C_g, c_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_g);
    cudaFree(B_g);
    cudaFree(C_g);
    ret[0] = (void*)C;
}

void call_gemm_nn_gpu(void **params, void **ret)
{
    int *AM = (int*)params[0];
    int *AN = (int*)params[1];
    int *BM = (int*)params[2];
    int *BN = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *A = (float*)params[5];
    float *B = (float*)params[6];
    float *C = (float*)params[7];
    float *A_g = NULL;
    float *B_g = NULL;
    float *C_g = NULL;
    cudaMalloc((void**)&A_g, AM[0]*AN[0]*sizeof(float));
    cudaMalloc((void**)&B_g, BM[0]*BN[0]*sizeof(float));
    cudaMalloc((void**)&C_g, AM[0]*BN[0]*sizeof(float));
    cudaMemcpy(A_g, A, AM[0]*AN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_g, B, BM[0]*BN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_g, C, AM[0]*BN[0]*sizeof(float), cudaMemcpyHostToDevice);
    gemm_nn_gpu(AM[0], AN[0], BM[0], BN[0], ALPHA[0], A_g, B_g, C_g);
    cudaMemcpy(C, C_g, AM[0]*BN[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_g);
    cudaFree(B_g);
    cudaFree(C_g);
    ret[0] = (void*)C;
}

void call_gemm_tn_gpu(void **params, void **ret)
{
    int *AM = (int*)params[0];
    int *AN = (int*)params[1];
    int *BM = (int*)params[2];
    int *BN = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *A = (float*)params[5];
    float *B = (float*)params[6];
    float *C = (float*)params[7];
    float *A_g = NULL;
    float *B_g = NULL;
    float *C_g = NULL;
    cudaMalloc((void**)&A_g, AM[0]*AN[0]*sizeof(float));
    cudaMalloc((void**)&B_g, BM[0]*BN[0]*sizeof(float));
    cudaMalloc((void**)&C_g, AM[0]*BN[0]*sizeof(float));
    cudaMemcpy(A_g, A, AM[0]*AN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_g, B, BM[0]*BN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_g, C, AN[0]*BN[0]*sizeof(float), cudaMemcpyHostToDevice);
    gemm_tn_gpu(AM[0], AN[0], BM[0], BN[0], ALPHA[0], A_g, B_g, C_g);
    cudaMemcpy(C, C_g, AN[0]*BN[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_g);
    cudaFree(B_g);
    cudaFree(C_g);
    ret[0] = (void*)C;
}

void call_gemm_nt_gpu(void **params, void **ret)
{
    int *AM = (int*)params[0];
    int *AN = (int*)params[1];
    int *BM = (int*)params[2];
    int *BN = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *A = (float*)params[5];
    float *B = (float*)params[6];
    float *C = (float*)params[7];
    float *A_g = NULL;
    float *B_g = NULL;
    float *C_g = NULL;
    cudaMalloc((void**)&A_g, AM[0]*AN[0]*sizeof(float));
    cudaMalloc((void**)&B_g, BM[0]*BN[0]*sizeof(float));
    cudaMalloc((void**)&C_g, AM[0]*BN[0]*sizeof(float));
    cudaMemcpy(A_g, A, AM[0]*AN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_g, B, BM[0]*BN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_g, C, AM[0]*BM[0]*sizeof(float), cudaMemcpyHostToDevice);
    gemm_nt_gpu(AM[0], AN[0], BM[0], BN[0], ALPHA[0], A_g, B_g, C_g);
    cudaMemcpy(C, C_g, AM[0]*BM[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_g);
    cudaFree(B_g);
    cudaFree(C_g);
    ret[0] = (void*)C;
}

void call_gemm_tt_gpu(void **params, void **ret)
{
    int *AM = (int*)params[0];
    int *AN = (int*)params[1];
    int *BM = (int*)params[2];
    int *BN = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *A = (float*)params[5];
    float *B = (float*)params[6];
    float *C = (float*)params[7];
    float *A_g = NULL;
    float *B_g = NULL;
    float *C_g = NULL;
    cudaMalloc((void**)&A_g, AM[0]*AN[0]*sizeof(float));
    cudaMalloc((void**)&B_g, BM[0]*BN[0]*sizeof(float));
    cudaMalloc((void**)&C_g, AM[0]*BN[0]*sizeof(float));
    cudaMemcpy(A_g, A, AM[0]*AN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_g, B, BM[0]*BN[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_g, C, AN[0]*BM[0]*sizeof(float), cudaMemcpyHostToDevice);
    gemm_tt_gpu(AM[0], AN[0], BM[0], BN[0], ALPHA[0], A_g, B_g, C_g);
    cudaMemcpy(C, C_g, AN[0]*BM[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_g);
    cudaFree(B_g);
    cudaFree(C_g);
    ret[0] = (void*)C;
}
