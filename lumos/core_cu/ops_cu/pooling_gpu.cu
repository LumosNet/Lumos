#include "pooling_gpu.h"

__global__ void avgpool_kernel(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
}

void avgpool_gpu(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space)
{

}

void maxpool_gpu(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index);

void avgpool_gradient_gpu(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n);
void maxpool_gradient_gpu(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n, int *index);
