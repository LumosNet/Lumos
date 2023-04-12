#include "softmax_gpu.h"

__global__ void softmax_kernel(float *data, int num, float *space, float *ALPHA)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = data[index] / ALPHA[0];
}

__global__ void softmax_grident_kernel(float *data, int num, float *space, float *ALPHA)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = (data[index] + ALPHA[0]) * data[index];
}

void softmax_gpu(float *data, int num, float *space, float *ALPHA)
{
    softmax_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space, ALPHA);
}

void softmax_grident_gpu(float *data, int num, float *space, float *ALPHA)
{
    softmax_grident_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space, ALPHA);
}

void softmax_exp_sum_gpu(float *data, int num, float *workspace, float *space)
{
    max_gpu(data, num, workspace+num);
    exp_list_gpu(data, num, workspace, workspace+num);
    sum_gpu(workspace, num, space);
}
