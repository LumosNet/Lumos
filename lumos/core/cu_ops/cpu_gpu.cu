#include "cpu_gpu.h"

__global__ void fill_kernel(float *data, int len, float x, int offset)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x)*offset;
    if (index >= len) return;
    data[index] = x;
}

void fill_gpu(float *data, int len, float x, int offset)
{
    fill_kernel<<<(len+BLOCK-1)/BLOCK, BLOCK>>>(data, len, x, offset);
}
