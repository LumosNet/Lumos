#include "connect_layer_gpu_call.h"

void call_forward_connect_layer_gpu(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *bias = (int*)params[1];
    char *active = (char*)params[2];
    int *h = (int*)params[3];
    int *w = (int*)params[4];
    int *c = (int*)params[5];
    int *num = (int*)params[6];
    float *input = (float*)params[7];
    float *output = (float*)params[8];
    float *kernel_weights = (float*)params[9];
    float *bias_weights = (float*)params[10];
    Layer *l = make_connect_layer(ksize[0], bias[0], active);
    l->initializegpu(l, w[0], h[0], c[0], num[0]);
    l->input = input;
    l->output = output;
    l->kernel_weights = kernel_weights;
    l->bias_weights = bias_weights;
    l->forwardgpu(*l, num[0]);
    ret[0] = l->output;
}

void call_backward_connect_layer_gpu(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *bias = (int*)params[1];
    char *active = (char*)params[2];
    int *h = (int*)params[3];
    int *w = (int*)params[4];
    int *c = (int*)params[5];
    float *rate = (float*)params[6];
    int *num = (int*)params[7];
    float *l_delta = (float*)params[8];
    float *n_delta = (float*)params[9];
    float *input = (float*)params[10];
    float *output = (float*)params[11];
    float *kernel_weights = (float*)params[12];
    float *update_kernel_weights = (float*)params[13];
    float *bias_weights = (float*)params[14];
    float *update_bias_weights = (float*)params[15];
    float *workspace = (float*)params[16];
    Layer *l = make_connect_layer(ksize[0], bias[0], active);
    l->initializegpu(l, w[0], h[0], c[0], num[0]);
    l->input = input;
    l->output = output;
    l->delta = l_delta;
    l->kernel_weights = kernel_weights;
    l->update_kernel_weights = update_kernel_weights;
    l->bias_weights = bias_weights;
    l->update_bias_weights = update_bias_weights;
    l->workspace = workspace;
    l->backwardgpu(*l, rate[0], num[0], n_delta);
    ret[0] = l->delta;
    ret[1] = l->update_kernel_weights;
    ret[2] = l->update_bias_weights;
}
