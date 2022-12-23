#include "bias_gpu_call.h"

void call_add_bias_gpu(void **params, void **ret)
{
    float *origin = (float*)params[0];
    float *bias = (float*)params[1];
    int *n = (int*)params[2];
    int *size = (int*)params[3];
    float *origin_gpu = NULL;
    float *bias_gpu = NULL;
    cudaMalloc((void**)&origin_gpu, size[0]*n[0]*sizeof(float));
    cudaMalloc((void**)&bias_gpu, n[0]*sizeof(float));
    cudaMemcpy(origin, origin_gpu, size[0]*n[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias, bias_gpu, n[0]*sizeof(float), cudaMemcpyHostToDevice);
    add_bias_gpu(origin_gpu, bias_gpu, n[0], size[0]);
    cudaMemcpy(origin_gpu, origin, size[0]*n[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(origin_gpu);
    cudaFree(bias_gpu);
    ret[0] = (void*)origin;
}
