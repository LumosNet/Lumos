#include "normalize_gpu.h"

void normalize_mean_gpu(float *data, int h, int w, int c, float *mean)
{
    int offset = h*w;
    for (int i = 0; i < c; ++i){
        float *data_c = data + i*offset;
        mean_gpu(data_c, offset, mean+i);
    }
}

void normalize_variance_gpu(float *data, int h, int w, int c, float *mean, float *variance)
{
    if (variance == NULL) printf("variance error\n");
    int offset = h*w;
    for (int i = 0; i < c; ++i){
        float *data_c = data + i*offset;
        variance_gpu(data_c, mean, offset, variance+i);
    }
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
    space_c[j] = (data_c[j] - mean[i]) / (sqrt(variance[i]) + .000001f);
}

void normalize_gpu(float *data, float *mean, float *variance, int h, int w, int c, float *space)
{
    int num = h*w*c;
    normalize_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, mean, variance, h, w, c, space);
}

__global__ void gradient_normalize_mean_kernel(float *beta, float *variance, int num, float *mean_delta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num) return;
    mean_delta[index] = (-1./sqrt(variance[index] + .00001f))*beta[index];
}

void gradient_normalize_mean_gpu(float *beta, float *variance, int num, float *mean_delta)
{
    gradient_normalize_mean_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(beta, variance, num, mean_delta);
}

__global__ void gradient_normalize_variance_kernel(float *beta, float *input, float *n_delta, float *mean, float *variance, int h, int w, int c, float *variance_delta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    variance_delta[index] = 0;
    for (int j = 0; j < h*w; ++j){
        variance_delta[index] += n_delta[index*h*w + j]*(input[index*h*w + j]-mean[index]);
    }
    variance_delta[index] *= -.5 * pow(variance[index] + .00001f, (float)(-3./2.)) * beta[index];
}

void gradient_normalize_variance_gpu(float *beta, float *input, float *n_delta, float *mean, float *variance, int h, int w, int c, float *variance_delta)
{
    gradient_normalize_variance_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(beta, input, n_delta, mean, variance, h, w, c, variance_delta);
}

__global__ void gradient_normalize_kernel(float *input, float *mean, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    space[index] = 0;
    for (int j = 0; j < h*w; ++j){
        l_delta[index*h*w + j] = n_delta[index*h*w + j] * mean_delta[index] + 2.0/(h*w)*(input[index*h*w + j]-mean[index])*variance_delta[index];
        space[index] += l_delta[index*h*w + j];
    }
}

void gradient_normalize_gpu(float *input, float *mean, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta, float *space)
{
    gradient_normalize_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(input, mean, mean_delta, variance_delta, h, w, c, n_delta, l_delta, space);
}

__global__ void  gradient_normalize_layer_kernel(int h, int w, int c, float *l_delta, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= c) return;
    for (int j = 0; j < h*w; ++j){
        l_delta[index*h*w + j] -= 1/(h*w)*space[index];
    }
}

void gradient_normalize_layer_gpu(int h, int w, int c, float *l_delta, float *space)
{
    gradient_normalize_layer_kernel<<<(c+BLOCK-1)/BLOCK, BLOCK>>>(h, w, c, l_delta, space);
}
