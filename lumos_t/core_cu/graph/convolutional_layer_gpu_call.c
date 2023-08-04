#include "convolutional_layer_gpu_call.h"

void call_forward_convolutional_layer_gpu(void **params, void **ret)
{
    int *filters = (int*)params[0];
    int *ksize = (int*)params[1];
    int *stride = (int*)params[2];
    int *pad = (int*)params[3];
    int *bias = (int*)params[4];
    int *normalization = (int*)params[5];
    char *active = (char*)params[6];
    int *h = (int*)params[8];
    int *w = (int*)params[9];
    int *c = (int*)params[10];
    int *num = (int*)params[11];
    float *input = (float*)params[12];
    float *output = (float*)params[13];
    float *kernel_weights = (float*)params[14];
    float *bias_weights = (float*)params[15];
    float *workspace = (float*)params[16];
    Layer *l = make_convolutional_layer(filters[0], ksize[0], stride[0], pad[0], bias[0], normalization[0], active);
    l->coretype = GPU;
    init_convolutional_layer(l, w[0], h[0], c[0]);
    l->input = input;
    l->output = output;
    l->kernel_weights_gpu = kernel_weights;
    l->bias_weights_gpu = bias_weights;
    l->workspace = workspace;
    l->forward(*l, num[0]);
    ret[0] = l->output;
}

void call_backward_convolutional_layer_gpu(void **params, void **ret)
{
    int *filters = (int*)params[0];
    int *ksize = (int*)params[1];
    int *stride = (int*)params[2];
    int *pad = (int*)params[3];
    int *bias = (int*)params[4];
    int *normalization = (int*)params[5];
    char *active = (char*)params[6];
    int *h = (int*)params[8];
    int *w = (int*)params[9];
    int *c = (int*)params[10];
    float *rate = (float*)params[11];
    int *num = (int*)params[12];
    float *l_delta = (float*)params[13];
    float *n_delta = (float*)params[14];
    float *input = (float*)params[15];
    float *output = (float*)params[16];
    float *kernel_weights = (float*)params[17];
    float *update_kernel_weights = (float*)params[18];
    float *bias_weights = (float*)params[19];
    float *update_bias_weights = (float*)params[20];
    float *workspace = (float*)params[21];
    Layer *l = make_convolutional_layer(filters[0], ksize[0], stride[0], pad[0], bias[0], normalization[0], active);
    l->coretype = GPU;
    init_convolutional_layer(l, w[0], h[0], c[0]);
    l->input = input;
    l->output = output;
    l->kernel_weights_gpu = kernel_weights;
    l->update_kernel_weights_gpu = update_kernel_weights;
    l->bias_weights_gpu = bias_weights;
    l->update_bias_weights_gpu = update_bias_weights;
    l->delta = l_delta;
    l->workspace = workspace;
    l->backward(*l, rate[0], num[0], n_delta);
    ret[0] = l->delta;
    ret[1] = l->update_kernel_weights_gpu;
    ret[2] = l->update_bias_weights_gpu;
}
