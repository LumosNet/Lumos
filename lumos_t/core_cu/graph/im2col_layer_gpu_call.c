#include "im2col_layer_gpu_call.h"

void call_forward_im2col_layer_gpu(void **params, void **ret)
{
    int *flag = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *num = (int*)params[4];
    float *input = (float*)params[5];
    float *output = (float*)params[6];
    Layer *l = make_im2col_layer(flag[0]);
    init_im2col_layer(l, w[0], h[0], c[0]);
    l->input = input;
    l->output = output;
    l->forward(*l, num[0]);
    ret[0] = output;
}

void call_backward_im2col_layer_gpu(void **params, void **ret)
{
    int *flag = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *num = (int*)params[4];
    float *rate = (float*)params[5];
    float *l_delta = (float*)params[6];
    float *n_delta = (float*)params[7];
    Layer *l = make_im2col_layer(flag[0]);
    init_im2col_layer(l, w[0], h[0], c[0]);
    l->delta = l_delta;
    l->backward(*l, rate[0], num[0], n_delta);
    ret[0] = l_delta;
}
