#include "convolution.h"

Image *convolutional(Image *img, Tensor *channel, int pad, int stride)
{
    Tensor *imgcol = im2col(img, channel->size[0], stride, pad);
    Tensor *convolutional = gemm(imgcol, channel);
    free_tensor(imgcol);
    return convolutional;
}