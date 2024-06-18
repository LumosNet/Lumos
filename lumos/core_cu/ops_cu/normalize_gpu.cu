#include "normalize_gpu.h"

__global__ void normalize_mean_kernel(float *data, int h, int w, int c, int subdivision, float *mean)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    mean[index] = 0;
    for (int j = 0; j < subdivision; ++j){
        for (int k = 0; k < h*w; ++k){
            mean[index] += data[j*h*w*c+index*h*w+k];
        }
    }
    mean[index] /= subdivision*h*w;
}

void normalize_mean_gpu(float *data, int h, int w, int c, int subdivision, float *mean)
{
    normalize_mean_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(data, h, w, c, subdivision, mean);
}

__global__ void normalize_variance_kernel(float *data, int h, int w, int c, int subdivision, float *mean, float *variance)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    variance[index] = 0;
    for (int j = 0; j < subdivision; ++j){
        for (int k = 0; k < h*w; ++k){
            variance[index] += pow(data[j*h*w*c+index*h*w+k]-mean[index], 2);
        }
    }
    variance[index] /= subdivision*h*w;
}

void normalize_variance_gpu(float *data, int h, int w, int c, int subdivision, float *mean, float *variance)
{
    normalize_variance_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(data, h, w, c, subdivision, mean, variance);
}

__global__ void normalize_kernel(float *data, float *mean, float *variance, int h, int w, int c, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= h*w*c) return;
    int offset = h*w;
    int i = index / (h*w);
    int j = index % (h*w);
    float *data_c = data + i*offset;
    float *space_c = space + i*offset;
    space_c[j] = (data_c[j] - mean[i]) / (sqrt(variance[i] + .00001f));
}

void normalize_gpu(float *data, float *mean, float *variance, int h, int w, int c, float *space)
{
    int num = h*w*c;
    normalize_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, mean, variance, h, w, c, space);
}

__global__ void gradient_normalize_mean_kernel(float *n_delta, float *variance, int h, int w, int c, float *mean_delta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    mean_delta[index] = 0;
    for (int j = 0; j < h*w; ++j){
        mean_delta[index] += n_delta[index*h*w+j];
    }
    mean_delta[index] *= (-1./sqrt(variance[index] + .00001f));
}

void gradient_normalize_mean_gpu(float *n_delta, float *variance, int h, int w, int c, float *mean_delta)
{
    gradient_normalize_mean_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(n_delta, variance, h, w, c, mean_delta);
}

__global__ void gradient_normalize_variance_kernel(float *n_delta, float *input, float *mean, float *variance, int h, int w, int c, float *variance_delta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    variance_delta[index] = 0;
    for (int j = 0; j < h*w; ++j){
        variance_delta[index] += n_delta[index*h*w+j]*(input[index*h*w+j]-mean[index]);
    }
    variance_delta[index] *= -.5 * pow(variance[index] + .00001f, (float)(-3./2.));
}

void gradient_normalize_variance_gpu(float *n_delta, float *input, float *mean, float *variance, int h, int w, int c, float *variance_delta)
{
    gradient_normalize_variance_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(n_delta, input, mean, variance, h, w, c, variance_delta);
}

__global__ void gradient_normalize_kernel(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    for (int j = 0; j < h*w; ++j){
        l_delta[index*h*w+j] = n_delta[index*h*w+j] * 1./(sqrt(variance[index] + .00001f)) + variance_delta[index] * 2. * (input[index*h*w+j] - mean[index]) / (h*w) + mean_delta[index]/(h*w);
    }
}

void gradient_normalize_gpu(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta)
{
    gradient_normalize_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(input, mean, variance, mean_delta, variance_delta, h, w, c, n_delta, l_delta);
}

__global__ void update_scale_kernel(float *output, float *delta, int h, int w, int c, float rate, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    float sum = 0;
    for (int j = 0; j < h*w; ++j){
        sum += output[index*h*w+j]*delta[index*h*w+j];
    }
    space[index] += rate * sum;
}

__global__ void update_bias_kernel(float *delta, int h, int w, int c, float rate, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    float sum = 0;
    for (int j = 0; j < h*w; ++j){
        sum += delta[index*h*w+j];
    }
    space[index] += rate * sum;
}

void update_scale_gpu(float *output, float *delta, int h, int w, int c, float rate, float *space)
{
    update_scale_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(output, delta, h, w, c, rate, space);
}

void update_bias_gpu(float *delta, int h, int w, int c, float rate, float *space)
{
    update_bias_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(delta, h, w, c, rate, space);
}
