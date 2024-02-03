#ifndef IM2COL_H
#define IM2COL_H

#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"

#ifdef __cplusplus
extern "C"{
#endif

void im2col(float *img, int h, int w, int c, int ksize, int stride, int pad, float *space);
void col2im(float *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, float *space);

#ifdef __cplusplus
}
#endif

#endif