#include "convolution.h"

Image *forward_convolutional(Image *img, Array *channel, int pad, int stride)
{
    int channels = 0;
    if (img->dim == 2) channels = 1;
    else if (img->dim) channels = img->size[2];
    else return NULL;
    int height_col = (img->size[1] + 2*pad - channel->size[0]) / stride + 1;
    int width_col = (img->size[0] + 2*pad - channel->size[0]) / stride + 1;
    Array *img2col = array_x(height_col*width_col, channel->size[0]*channel->size[0]*channels, 0);
    im2col(img, channel->size[0], stride, pad, img2col->data);
    Array *k = copy(channel);
    resize_ar(k, channel->num, 1);
    Array *convolutional = gemm(img2col, k);
    int size[] = {width_col, height_col, 1};
    resize(convolutional, 3, size);
    del(k);
    return convolutional;
}