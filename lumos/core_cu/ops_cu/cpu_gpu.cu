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

__global__ void matrix_add_kernel(float *data_a, float *data_b, int num, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    printf("%d\n", index);
    space[index] = data_a[index] + data_b[index];
}

__global__ void matrix_subtract_kernel(float *data_a, float *data_b, int num, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = data_a[index] - data_b[index];
}

__global__ void matrix_multiply_kernel(float *data_a, float *data_b, int num, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = data_a[index] * data_b[index];
}

__global__ void matrix_divide_kernel(float *data_a, float *data_b, int num, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = data_a[index] / data_b[index];
}

__global__ void saxpy_kernel(float *data_a, float *data_b, int num, float x, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = data_a[index] + x * data_b[index];
}

void matrix_add_gpu(float *data_a, float *data_b, int num, float *space)
{
    matrix_add_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data_a, data_b, num, space);
}

void matrix_subtract_gpu(float *data_a, float *data_b, int num, float *space)
{
    matrix_subtract_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data_a, data_b, num, space);
}

void matrix_multiply_gpu(float *data_a, float *data_b, int num, float *space)
{
    matrix_multiply_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data_a, data_b, num, space);
}

void matrix_divide_gpu(float *data_a, float *data_b, int num, float *space)
{
    matrix_divide_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data_a, data_b, num, space);
}

void saxpy_gpu(float *data_a, float *data_b, int num, float x, float *space)
{
    saxpy_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data_a, data_b, num, x, space);
}

__global__ void sum_channel_kernel(float *data, int h, int w, int c, float ALPHA, float *space)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= c) return;
    float sum = 0;
    int offset = k*h*w;
    for (int i = 0; i < h; ++i){
        for (int j = 0; j < w; ++j){
            sum += data[offset + i*w + j] * ALPHA;
        }
    }
    space[k] = sum;
}

void sum_channel_gpu(float *data, int h, int w, int c, float ALPHA, float *space)
{
    sum_channel_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(data, h, w, c, ALPHA, space);
}
