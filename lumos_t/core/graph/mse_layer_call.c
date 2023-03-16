#include "mse_layer_call.h"

void call_forward_mse_layer(void **params, void **ret)
{
    int *group = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *num = (int*)params[4];
    float *input = (float*)params[5];
    float *output = (float*)params[6];
    float *truth = (float*)params[7];
    float *workspace = (float*)params[8];
    float *loss = (float*)params[9];
    Layer *l = make_mse_layer(group[0]);
    init_mse_layer(l, w[0], h[0], c[0]);
    l->input = input;
    l->output = output;
    l->truth = truth;
    l->workspace = workspace;
    l->loss = loss;
    l->forward(*l, num[0]);
    ret[0] = l->output;
    ret[1] = l->loss;
}

void call_backward_mse_layer(void **params, void **ret)
{

}
