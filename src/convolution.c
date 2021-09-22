#include "convolution.h"

Image *forward_conv(Image *img, Array *channel, int pad, int stride)
{
    int height_col = (img->size[1] + 2*pad - channel->size[0]) / stride + 1;
    int width_col = (img->size[0] + 2*pad - channel->size[0]) / stride + 1;
    Array *img2col = im2col(img, channel->size[0], stride, pad);
    Array *k = copy(channel);
    resize_ar(k, channel->num, 1);
    Array *convolutional = gemm(img2col, k);
    int size[] = {width_col, height_col, 1};
    resize(convolutional, 3, size);
    del(k);
    return convolutional;
}