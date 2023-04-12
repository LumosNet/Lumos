#include "softmax_gpu.h"

__global__ void softmax_kernel(float *data, int num, float *space)
{

}

__global__ void softmax_grident_kernel(float *data, int num, float *space)
{

}

void softmax_gpu(float *data, int num, float *space);
void softmax_grident_gpu(float *data, int num, float *space);
