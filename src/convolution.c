#include "convolution.h"

Image *convolutional(Image *img, Array *channel, int pad, int stride)
{
    Array *imgcol = im2col(img, channel->size[0], stride, pad);
    Array *convolutional = gemm(imgcol, channel);
    del(imgcol);
    return convolutional;
}