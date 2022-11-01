#include "bias_gpu.h"

__global__ void add_bias_kernel(float *origin, float *bias, int n, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= size) return;
    origin[i * size + j] += bias[i];
}

void add_bias_gpu(float *origin, float *bias, int n, int size)
{
    dim3 dimGrad((n+BLOCK-1)/BLOCK, (size+BLOCK-1)/BLOCK);
    dim3 dimBlock(BLOCK, BLOCK);
    add_bias_kernel<<<dimGrad, dimBlock>>>(origin, bias, n, size);
}
