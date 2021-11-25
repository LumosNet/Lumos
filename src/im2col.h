#ifndef IM2COL_H
#define IM2COL_H

#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "array.h"

#ifdef __cplusplus
extern "C"{
#endif

Tensor *im2col(Image *img, int ksize, int stride, int pad);
Tensor *col2im(Image *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c);

#ifdef __cplusplus
}
#endif

#endif