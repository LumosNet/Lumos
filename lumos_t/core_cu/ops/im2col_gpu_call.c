#include "im2col_gpu_call.h"

void call_im2col_gpu(void **params, void **ret)
{
    float *img = (float*)params[0];
    int *height = (int*)params[1];
    int *width = (int*)params[2];
    int *channel = (int*)params[3];
    int *ksize = (int*)params[4];
    int *stride = (int*)params[5];
    int *pad = (int*)params[6];
    float *space = (float*)params[7];
    float *img_g = NULL;
    float *space_g = NULL;
    int height_col = (height[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    int width_col = (width[0] + 2 * pad[0] - ksize[0]) / stride[0] + 1;
    int channels_col = channel[0] * ksize[0] * ksize[0];
    cudaMalloc((void**)&img_g, height[0]*width[0]*channel[0]*sizeof(float));
    cudaMalloc((void**)&space_g, height_col*width_col*channels_col*sizeof(float));
    cudaMemcpy(img_g, img, height[0]*width[0]*channel[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, height_col*width_col*channels_col*sizeof(float), cudaMemcpyHostToDevice);
    im2col_gpu(img_g, height[0], width[0], channel[0], ksize[0], stride[0], pad[0], space_g);
    cudaMemcpy(space, space_g, height_col*width_col*channels_col*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(img_g);
    cudaFree(space_g);
    ret[0] = (void*)space;
}

void call_col2im_gpu(void **params, void **ret)
{
    float *img = (float*)params[0];
    int *ksize = (int*)params[1];
    int *stride = (int*)params[2];
    int *pad = (int*)params[3];
    int *out_h = (int*)params[4];
    int *out_w = (int*)params[5];
    int *out_c = (int*)params[6];
    float *space = (float*)params[7];
    col2im_gpu(img, ksize[0], stride[0], pad[0], out_h[0], out_w[0], out_c[0], space);
    ret[0] = (void*)space;
}
