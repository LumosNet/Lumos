#include "pooling_gpu.h"

__global__ void avgpool_kernel(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    int k = index / (out_h*out_w);
    int i = (index % (out_h*out_w)) / out_w;
    int j = (index % (out_h*out_w)) % out_w;
    if (k >= c || i >= out_h || j >= out_w) return;
    int x = i*stride;
    int y = j*stride;
    float temp = 0;
    for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
        for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
            int index_i = x + ksize_i - pad;
            int index_j = y + ksize_j - pad;
            if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
            temp += im[k*h*w + index_i*w + index_j];
        }
    }
    space[k*out_h*out_w + i*out_w + j] = temp / (ksize*ksize);
}

void avgpool_gpu(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    avgpool_kernel<<<(out_h*out_w*c + BLOCK - 1)/BLOCK, BLOCK>>>(im, h, w, c, ksize, stride, pad, space);
}

__global__ void maxpool_kernel(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index)
{
    int index_g = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    int k = index_g / (out_h*out_w);
    int i = (index_g % (out_h*out_w)) / out_w;
    int j = (index_g % (out_h*out_w)) % out_w;
    if (k >= c || i >= out_h || j >= out_w) return;
    int x = i*stride;
    int y = j*stride;
    int max_index = -1;
    float max = -99;
    for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
        for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
            int index_i = x + ksize_i - pad;
            int index_j = y + ksize_j - pad;
            if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
            if (im[k*h*w + index_i*w + index_j] > max){
                max = im[k*h*w + index_i*w + index_j];
                max_index = k*h*w + index_i*w + index_j;
            }
        }
    }
    if (max_index == -1) max = 0;
    space[k*out_h*out_w + i*out_w + j] = max;
    index[k*out_h*out_w + i*out_w + j] = max_index;
}

void maxpool_gpu(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    maxpool_kernel<<<(out_h*out_w*c + BLOCK - 1)/BLOCK, BLOCK>>>(im, h, w, c, ksize, stride, pad, space, index);
}

__global__ void avgpool_gradient_kernel(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    int k = index / (out_h*out_w);
    int i = (index % (out_h*out_w)) / out_w;
    int j = (index % (out_h*out_w)) % out_w;
    if (k >= c || i >= out_h || j >= out_w) return;
    int x = i*stride;
    int y = j*stride;
    for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
        for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
            int index_i = x + ksize_i - pad;
            int index_j = y + ksize_j - pad;
            if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
            delta_l[k*h*w + index_i*w + index_j] += delta_n[k*out_h*out_w + i*out_w + j] / (ksize*ksize);
        }
    }
}

void avgpool_gradient_gpu(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    fill_gpu(delta_l, h*w*c, 0, 1);
    avgpool_gradient_kernel<<<(out_h*out_w*c + BLOCK - 1)/BLOCK, BLOCK>>>(delta_l, h, w, c, ksize, stride, pad, delta_n);
}

__global__ void maxpool_gradient_kernel(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n, int *index)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_h*out_w*c || index[i] == -1) return;
    delta_l[index[i]] += delta_n[i];
}

void maxpool_gradient_gpu(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n, int *index)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    fill_gpu(delta_l, h*w*c, 0, 1);
    maxpool_gradient_kernel<<<(out_h*out_w*c + BLOCK - 1)/BLOCK, BLOCK>>>(delta_l, h, w, c, ksize, stride, pad, delta_n, index);
}
