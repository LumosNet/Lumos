#include "bias_gpu.h"

__global__ void add_bias_kernel(float *origin, float *bias, int n, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n*size) return;
    int bias_index = index/size;
    origin[index] += bias[bias_index];
}

__global__ void scale_bias_kernel(float *origin, float *bias, int n, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n*size) return;
    int bias_index = index/size;
    origin[index] *= bias[bias_index];
}

void add_bias_gpu(float *origin, float *bias, int n, int size)
{
    int num = n*size;
    add_bias_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(origin, bias, n, size);
}

void scale_bias_gpu(float *origin, float *bias, int n, int size)
{
    int num = n*size;
    scale_bias_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(origin, bias, n, size);
}
