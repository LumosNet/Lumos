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

__global__ void min_kernel(float *data, int num, float *space)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tmp[BLOCK];
    if(gid < num)
    {
        tmp[threadIdx.x] = data[gid];
    }
    else
    {
        tmp[threadIdx.x] = MAXFLOAT;
    }
    __syncthreads();

    for(int strip = blockDim.x / 2; strip > 0; strip = strip / 2)
    {
        if(threadIdx.x < strip)
            tmp[threadIdx.x] = (tmp[threadIdx.x + strip] < tmp[threadIdx.x]) ? tmp[threadIdx.x + strip] : tmp[threadIdx.x];
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        space[blockIdx.x] = tmp[0];
    }
}

__global__ void max_kernel(float *data, int num, float *space)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tmp[BLOCK];
    if(gid < num)
    {
        tmp[threadIdx.x] = data[gid];
    }
    else
    {
        tmp[threadIdx.x] = -MAXFLOAT;
    }
    __syncthreads();

    for(int strip = blockDim.x / 2; strip > 0; strip = strip / 2)
    {
        if(threadIdx.x < strip)
            tmp[threadIdx.x] = (tmp[threadIdx.x + strip] > tmp[threadIdx.x]) ? tmp[threadIdx.x + strip] : tmp[threadIdx.x];
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        space[blockIdx.x] = tmp[0];
    }
}

__global__ void sum_kernel(float *data, int num, float *space)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tmp[BLOCK];
    if(gid < num)
    {
        tmp[threadIdx.x] = data[gid];
    }
    else
    {
        tmp[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for(int strip = blockDim.x / 2; strip > 0; strip = strip / 2)
    {
        if(threadIdx.x < strip)
            tmp[threadIdx.x] += tmp[threadIdx.x + strip];
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        space[blockIdx.x] = tmp[0];
    }
}

__global__ void mean_kernel(float *data, int num, float *space)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float scale = 1. / num;
    __shared__ float tmp[BLOCK];
    if(gid < num)
    {
        tmp[threadIdx.x] = data[gid];
    }
    else
    {
        tmp[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for(int strip = blockDim.x / 2; strip > 0; strip = strip / 2)
    {
        if(threadIdx.x < strip)
            tmp[threadIdx.x] += tmp[threadIdx.x + strip];
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        space[blockIdx.x] = tmp[0] * scale;
    }
}

__global__ void  variance_kernel(float *data, float *mean, int num, float *variance)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i;
    for(i = 0; i < num; i += threads){
        int index = filter*num + i + id;
        local[id] += (i+id < num) ? powf((data[index] - mean[0]), 2) : 0;
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= num;
    }
}

void min_gpu(float *data, int num, float *space)
{
    min_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space);
}

void max_gpu(float *data, int num, float *space)
{
    max_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space);
}

void sum_gpu(float *data, int num, float *space)
{
    sum_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space);
}

void mean_gpu(float *data, int num, float *space)
{
    mean_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space);
}

void variance_gpu(float *data, float *mean, int num, float *space)
{
    variance_kernel<<<1, BLOCK>>>(data, mean, num, space);
}

__global__ void exp_list_kernel(float *data, int num, float *space, float *ALPHA)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = exp(data[index]-ALPHA[0]);
}

void exp_list_gpu(float *data, int num, float *space, float *ALPHA)
{
    exp_list_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space, ALPHA);
}
