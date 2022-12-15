#include "bias_call.h"

void call_add_bias(void **params, void **ret)
{
    float *origin = (float*)params[0];
    float *bias = (float*)params[1];
    int *n = (int*)params[2];
    int *size = (int*)params[3];
    add_bias(origin, bias, n[0], size[0]);
    ret[0] = (void*)origin;
}
