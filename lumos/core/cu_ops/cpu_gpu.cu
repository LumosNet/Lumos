#include "cpu_gpu.h"

__global__ void fill_kernel(float *data, int len, float x, int offset)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x)*offset;
    if (index >= len) return;
    data[index] = x;
}

__global__ void multy_kernel(float *data, int len, float x, int offset)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x)*offset;
    if (index >= len) return;
    data[index] *= x;
}

__global__ void add_kernel(float *data, int len, float x, int offset)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x)*offset;
    if (index >= len) return;
    data[index] += x;
}

void fill_gpu(float *data, int len, float x, int offset)
{
    fill_kernel<<<(len+BLOCK-1)/BLOCK, BLOCK>>>(data, len, x, offset);
}

void multy_gpu(float *data, int len, float x, int offset)
{
    multy_kernel<<<(len+BLOCK-1)/BLOCK, BLOCK>>>(data, len, x, offset);
}

void add_gpu(float *data, int len, float x, int offset)
{
    add_kernel<<<(len+BLOCK-1)/BLOCK, BLOCK>>>(data, len, x, offset);
}

void min_gpu(float *data, int num, float *space);
void max_gpu(float *data, int num, float *space);
void sum_gpu(float *data, int num, float *space);
void mean_gpu(float *data, int num, float *space);

void matrix_add_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_subtract_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_multiply_gpu(float *data_a, float *data_b, int num, float *space);
void matrix_divide_gpu(float *data_a, float *data_b, int num, float *space);

void saxpy_gpu(float *data_a, float *data_b, int num, float x, float *space);
