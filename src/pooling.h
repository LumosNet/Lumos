#ifndef POOLING_H
#define POOLING_H

#include "lumos.h"
#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_avg_pool(float *img, int h, int w, int c, int ksize);
void forward_max_pool(float *img, int h, int w, int c, int ksize, int *index);

void backward_avg_pool(Tensor *img, int ksize, int height, int width);
void backward_max_pool(Tensor *img, int ksize, int height, int width, int *index);

#ifdef __cplusplus
}
#endif

#endif