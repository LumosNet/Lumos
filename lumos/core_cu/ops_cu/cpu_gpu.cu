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

__global__ void  mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1.f/(batch * spatial);
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

__global__ void variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1.f/(batch * spatial - 1);
    int j,k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += powf((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

__global__ void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? powf((x[index] - mean[filter]), 2) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    fast_mean_kernel<<<filters, BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    fast_variance_kernel<<<filters, BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}


extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}
