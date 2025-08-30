#include "bias_gpu_call.h"

void call_add_bias_gpu(void **params, void **ret)
{
    float *origin = (float*)params[0];
    float *bias = (float*)params[1];
    int *n = (int*)params[2];
    int *size = (int*)params[3];
    add_bias_gpu(origin, bias, n[0], size[0]);
    ret[0] = (void*)origin;
    ret[1] = (void*)bias;
}

void call_scale_bias_gpu(void **params, void **ret)
{
    float *origin = (float*)params[0];
    float *bias = (float*)params[1];
    int *n = (int*)params[2];
    int *size = (int*)params[3];
    scale_bias_gpu(origin, bias, n[0], size[0]);
    ret[0] = (void*)origin;
    ret[1] = (void*)bias;
}
