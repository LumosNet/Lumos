#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "lumos.h"
#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

Image *convolutional(Image *img, Array *channel, int pad, int stride);

#ifdef __cplusplus
}
#endif

#endif