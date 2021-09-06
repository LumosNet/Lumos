#ifndef IM2COL_H
#define IM2COL_H

float get_pixel(float *im, int row, int col, int channel, int h, int w, int c, int pad);
void im2col_cpu(float *im, int h, int w, int c, int size, int stride, int pad, float *output);
#endif