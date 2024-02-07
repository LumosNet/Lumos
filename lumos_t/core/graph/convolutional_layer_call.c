#include "convolutional_layer_call.h"

void call_forward_convolutional_layer(void **params, void **ret)
{
    int *filters = (int*)params[0];
    int *ksize = (int*)params[1];
    int *stride = (int*)params[2];
    int *pad = (int*)params[3];
    int *bias = (int*)params[4];
    char *active = (char*)params[5];
    int *h = (int*)params[6];
    int *w = (int*)params[7];
    int *c = (int*)params[8];
    int *num = (int*)params[9];
    float *input = (float*)params[10];
    float *output = (float*)params[11];
    float *kernel_weights = (float*)params[12];
    float *bias_weights = (float*)params[13];
    float *workspace = (float*)params[14];
    Layer *l = make_convolutional_layer(filters[0], ksize[0], stride[0], pad[0], bias[0], active);
    init_convolutional_layer(l, w[0], h[0], c[0], num[0]);
    l->input = input;
    l->output = output;
    l->kernel_weights = kernel_weights;
    l->bias_weights = bias_weights;
    l->workspace = workspace;
    l->forward(*l, num[0]);
    ret[0] = l->output;
}

void call_backward_convolutional_layer(void **params, void **ret)
{
    int *filters = (int*)params[0];
    int *ksize = (int*)params[1];
    int *stride = (int*)params[2];
    int *pad = (int*)params[3];
    int *bias = (int*)params[4];
    char *active = (char*)params[5];
    int *h = (int*)params[6];
    int *w = (int*)params[7];
    int *c = (int*)params[8];
    float *rate = (float*)params[9];
    int *num = (int*)params[10];
    float *l_delta = (float*)params[11];
    float *n_delta = (float*)params[12];
    float *input = (float*)params[13];
    float *output = (float*)params[14];
    float *kernel_weights = (float*)params[15];
    float *update_kernel_weights = (float*)params[16];
    float *bias_weights = (float*)params[17];
    float *update_bias_weights = (float*)params[18];
    float *workspace = (float*)params[19];
    Layer *l = make_convolutional_layer(filters[0], ksize[0], stride[0], pad[0], bias[0], active);
    init_convolutional_layer(l, w[0], h[0], c[0], num[0]);
    l->input = input;
    l->output = output;
    l->kernel_weights = kernel_weights;
    l->update_kernel_weights = update_kernel_weights;
    l->bias_weights = bias_weights;
    l->update_bias_weights = update_bias_weights;
    l->delta = l_delta;
    l->workspace = workspace;
    l->backward(*l, rate[0], num[0], n_delta);
    ret[0] = l->delta;
    ret[1] = l->update_kernel_weights;
    ret[2] = l->update_bias_weights;
}
