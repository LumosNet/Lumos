#include "maxpool_layer_gpu_call.h"

void call_forward_maxpool_layer_gpu(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *num = (int*)params[4];
    float *input = (float*)params[5];
    float *output = (float*)params[6];
    int *index = (int*)params[7];
    Layer *l = make_maxpool_layer(ksize[0], ksize[0], 0);
    l->coretype = GPU;
    init_maxpool_layer(l, w[0], h[0], c[0]);
    l->input = input;
    l->output = output;
    l->maxpool_index = index;
    l->forward(*l, num[0]);
    ret[0] = l->output;
    ret[1] = l->maxpool_index;
}

void call_backward_maxpool_layer_gpu(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    float *rate = (float*)params[4];
    int *num = (int*)params[5];
    float *l_delta = (float*)params[6];
    float *n_delta = (float*)params[7];
    int *index = (int*)params[8];
    Layer *l = make_maxpool_layer(ksize[0], ksize[0], 0);
    l->coretype = GPU;
    init_maxpool_layer(l, w[0], h[0], c[0]);
    l->delta = l_delta;
    l->maxpool_index = index;
    l->backward(*l, rate[0], num[0], n_delta);
    ret[0] = l->delta;
}
