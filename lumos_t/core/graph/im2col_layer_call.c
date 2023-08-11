#include "im2col_layer_call.h"

void call_forward_im2col_layer(void **params, void **ret)
{
    int *h = (int*)params[0];
    int *w = (int*)params[1];
    int *c = (int*)params[2];
    int *num = (int*)params[3];
    float *input = (float*)params[4];
    float *output = (float*)params[5];
    Layer *l = make_im2col_layer();
    l->coretype = CPU;
    init_im2col_layer(l, w[0], h[0], c[0]);
    l->input = input;
    l->output = output;
    l->forward(*l, num[0]);
    ret[0] = l->output;
    ret[1] = &l->output_h;
    ret[2] = &l->output_w;
    ret[3] = &l->output_c;
}

void call_backward_im2col_layer(void **params, void **ret)
{
    int *h = (int*)params[0];
    int *w = (int*)params[1];
    int *c = (int*)params[2];
    int *num = (int*)params[3];
    float *rate = (float*)params[4];
    float *l_delta = (float*)params[5];
    float *n_delta = (float*)params[6];
    Layer *l = make_im2col_layer();
    l->coretype = CPU;
    init_im2col_layer(l, w[0], h[0], c[0]);
    l->delta = l_delta;
    l->backward(*l, rate[0], num[0], n_delta);
    ret[0] = l->delta;
}
