#include "pooling_gpu_call.h"

void call_avgpool_gpu(void **params, void **ret)
{
    float *im = (float*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *ksize = (int*)params[4];
    int *stride = (int*)params[5];
    int *pad = (int*)params[6];
    float *space = (float*)params[7];
    int out_h = (h[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    int out_w = (w[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    float *im_g = NULL;
    float *space_g = NULL;
    cudaMalloc((void**)&im_g, h[0]*w[0]*c[0]*sizeof(float));
    cudaMalloc((void**)&space_g, out_h*out_w*c[0]*sizeof(float));
    cudaMemcpy(im_g, im, h[0]*w[0]*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, out_h*out_w*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    avgpool_gpu(im_g, h[0], w[0], c[0], ksize[0], stride[0], pad[0], space_g);
    cudaMemcpy(space, space_g, out_h*out_w*c[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(im_g);
    cudaFree(space_g);
    ret[0] = (void*)space;
}

void call_maxpool_gpu(void **params, void **ret)
{
    float *im = (float*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *ksize = (int*)params[4];
    int *stride = (int*)params[5];
    int *pad = (int*)params[6];
    float *space = (float*)params[7];
    int *index = (int*)params[8];
    int out_h = (h[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    int out_w = (w[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    float *im_g = NULL;
    float *space_g = NULL;
    int *index_g = NULL;
    cudaMalloc((void**)&im_g, h[0]*w[0]*c[0]*sizeof(float));
    cudaMalloc((void**)&space_g, out_h*out_w*c[0]*sizeof(float));
    cudaMalloc((void**)&index_g, out_h*out_w*c[0]*sizeof(int));
    cudaMemcpy(im_g, im, h[0]*w[0]*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, out_h*out_w*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(index_g, index, out_h*out_w*c[0]*sizeof(int), cudaMemcpyHostToDevice);
    maxpool_gpu(im_g, h[0], w[0], c[0], ksize[0], stride[0], pad[0], space_g, index_g);
    cudaMemcpy(space, space_g, out_h*out_w*c[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(index, index_g, out_h*out_w*c[0]*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(im_g);
    cudaFree(space_g);
    cudaFree(index_g);
    ret[0] = (void*)space;
    ret[1] = (void*)index;
}

void call_avgpool_gradient_gpu(void **params, void **ret)
{
    float *delta_l = (float*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *ksize = (int*)params[4];
    int *stride = (int*)params[5];
    int *pad = (int*)params[6];
    float *delta_n = (float*)params[7];
    int out_h = (h[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    int out_w = (w[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    float *delta_l_g = NULL;
    float *delta_n_g = NULL;
    cudaMalloc((void**)&delta_l_g, h[0]*w[0]*c[0]*sizeof(float));
    cudaMalloc((void**)&delta_n_g, out_h*out_w*c[0]*sizeof(float));
    cudaMemcpy(delta_l_g, delta_l, h[0]*w[0]*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_n_g, delta_n, out_h*out_w*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    avgpool_gradient_gpu(delta_l_g, h[0], w[0], c[0], ksize[0], stride[0], pad[0], delta_n_g);
    cudaMemcpy(delta_l, delta_l_g, h[0]*w[0]*c[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(delta_l_g);
    cudaFree(delta_n_g);
    ret[0] = (void*)delta_l;
}

void call_maxpool_gradient_gpu(void **params, void **ret)
{
    float *delta_l = (float*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *ksize = (int*)params[4];
    int *stride = (int*)params[5];
    int *pad = (int*)params[6];
    float *delta_n = (float*)params[7];
    int *index = (int*)params[8];
    int out_h = (h[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    int out_w = (w[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    float *delta_l_g = NULL;
    float *delta_n_g = NULL;
    int *index_g = NULL;
    cudaMalloc((void**)&delta_l_g, h[0]*w[0]*c[0]*sizeof(float));
    cudaMalloc((void**)&delta_n_g, out_h*out_w*c[0]*sizeof(float));
    cudaMalloc((void**)&index_g, out_h*out_w*c[0]*sizeof(int));
    cudaMemcpy(delta_l_g, delta_l, h[0]*w[0]*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_n_g, delta_n, out_h*out_w*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(index_g, index, out_h*out_w*c[0]*sizeof(int), cudaMemcpyHostToDevice);
    maxpool_gradient_gpu(delta_l_g, h[0], w[0], c[0], ksize[0], stride[0], pad[0], delta_n_g, index_g);
    cudaMemcpy(delta_l, delta_l_g, h[0]*w[0]*c[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(delta_l_g);
    cudaFree(delta_n_g);
    cudaFree(index_g);
    ret[0] = (void*)delta_l;
}
