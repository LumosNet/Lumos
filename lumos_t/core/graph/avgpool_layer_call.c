#include "avgpool_layer_call.h"

void call_forward_avgpool_layer(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *num = (int*)params[4];
    float *input = (float*)params[5];
    float *output = (float*)params[6];
    Layer *l = make_avgpool_layer(ksize[0], ksize[0], 0);
    init_avgpool_layer(l, w[0], h[0], c[0], num[0]);
    l->input = input;
    l->output = output;
    l->forward(*l, num[0]);
    ret[0] = l->output;
}

void call_backward_avgpool_layer(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    float *rate = (float*)params[4];
    int *num = (int*)params[5];
    float *n_delta = (float*)params[6];
    float *l_delta = (float*)params[7];
    Layer *l = make_avgpool_layer(ksize[0], ksize[0], 0);
    init_avgpool_layer(l, w[0], h[0], c[0], num[0]);
    l->delta = l_delta;
    l->backward(*l, rate[0], num[0], n_delta);
    ret[0] = l->delta;
}
