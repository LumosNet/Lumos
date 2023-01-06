#include "im2col_call.h"

void call_im2col(void **params, void **ret)
{
    float *img = (float*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *ksize = (int*)params[4];
    int *stride = (int*)params[5];
    int *pad = (int*)params[6];
    float *space = (float*)params[7];
    im2col(img, h[0], w[0], c[0], ksize[0], stride[0], pad[0], space);
    ret[0] = (void*)space;
}

void call_col2im(void **params, void **ret)
{
    float *img = (float*)params[0];
    int *ksize = (int*)params[1];
    int *stride = (int*)params[2];
    int *pad = (int*)params[3];
    int *out_h = (int*)params[4];
    int *out_w = (int*)params[5];
    int *out_c = (int*)params[6];
    float *space = (float*)params[7];
    col2im(img, ksize[0], stride[0], pad[0], out_h[0], out_w[0], out_c[0], space);
    ret[0] = (void*)space;
}
