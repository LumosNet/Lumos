#ifndef IM2COL_H
#define IM2COL_H

#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "array.h"

#ifdef __cplusplus
extern "C"{
#endif

void im2col(Tensor *img, int ksize, int stride, int pad, float *space);
void col2im(Tensor *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, float *space);

#ifdef __cplusplus
}
#endif

#endif