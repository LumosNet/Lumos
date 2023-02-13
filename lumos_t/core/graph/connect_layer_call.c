#include "connect_layer_call.h"

void call_forward_connect_layer(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *bias = (int*)params[1];
    char *active = (char*)params[2];
    char *weights_init = (char*)params[3];
    int *h = (int*)params[4];
    int *w = (int*)params[5];
    int *c = (int*)params[6];
    int *num = (int*)params[7];
    float *input = (float*)params[8];
    float *output = (float*)params[9];
    float *kernel_weights = (float*)params[10];
    float *bias_weights = (float*)params[11];
    Layer *l = make_connect_layer(ksize[0], bias[0], active, weights_init);
    init_connect_layer(l, w[0], h[0], c[0]);
    l->input = input;
    l->output = output;
    l->kernel_weights = kernel_weights;
    l->bias_weights = bias_weights;
    l->forward(*l, num[0]);
    ret[0] = l->output;
}

void call_backward_connect_layer(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *bias = (int*)params[1];
    char *active = (char*)params[2];
    char *weights_init = (char*)params[3];
    int *h = (int*)params[4];
    int *w = (int*)params[5];
    int *c = (int*)params[6];
    float *rate = (float*)params[7];
    int *num = (int*)params[8];
    float *l_delta = (float*)params[9];
    float *n_delta = (float*)params[10];
    float *input = (float*)params[11];
    float *output = (float*)params[12];
    float *kernel_weights = (float*)params[13];
    float *update_kernel_weights = (float*)params[14];
    float *bias_weights = (float*)params[15];
    float *update_bias_weights = (float*)params[16];
    float *workspace = (float*)params[17];
    Layer *l = make_connect_layer(ksize[0], bias[0], active, weights_init);
    init_connect_layer(l, w[0], h[0], c[0]);
    l->input = input;
    l->output = output;
    l->delta = l_delta;
    l->kernel_weights = kernel_weights;
    l->update_kernel_weights = update_kernel_weights;
    l->bias_weights = bias_weights;
    l->update_bias_weights = update_bias_weights;
    l->workspace = workspace;
    l->backward(*l, rate[0], num[0], n_delta);
    ret[0] = l->delta;
    ret[1] = l->update_kernel_weights;
    ret[2] = l->update_bias_weights;
}
