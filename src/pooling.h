#ifndef POOLING_H
#define POOLING_H

#include "lumos.h"
#include "image.h"
#include "convolution.h"

#ifdef __cplusplus
extern "C" {
#endif

Image *forward_avg_pool(Image *img, int ksize);
Image *forward_max_pool(Image *img, int ksize, int *index);

Image *backward_avg_pool(Image *img, int ksize, int height, int width);
Image *backward_max_pool(Image *img, int ksize, int height, int width, int *index);

#ifdef __cplusplus
}
#endif

#endif