#ifndef POOLING_H
#define POOLING_H

#include "lumos.h"
#include "image.h"
#include "convolution.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor *forward_avg_pool(Tensor *img, int ksize);
Tensor *forward_max_pool(Tensor *img, int ksize, int *index);

Tensor *backward_avg_pool(Tensor *img, int ksize, int height, int width);
Tensor *backward_max_pool(Tensor *img, int ksize, int height, int width, int *index);

#ifdef __cplusplus
}
#endif

#endif