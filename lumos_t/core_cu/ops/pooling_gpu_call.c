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
    avgpool_gpu(im, h[0], w[0], c[0], ksize[0], stride[0], pad[0], space);
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
    maxpool_gpu(im, h[0], w[0], c[0], ksize[0], stride[0], pad[0], space, index);
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
    avgpool_gradient_gpu(delta_l, h[0], w[0], c[0], ksize[0], stride[0], pad[0], delta_n);
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
    maxpool_gradient_gpu(delta_l, h[0], w[0], c[0], ksize[0], stride[0], pad[0], delta_n, index);
    ret[0] = (void*)delta_l;
}
