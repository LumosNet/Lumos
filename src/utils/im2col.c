#include "im2col.h"

float get_pixel(float *im, int row, int col, int channel, int h, int w, int c, int pad)
{
    row -= pad;
    col -= pad;
    if (row < 0 || row >= h || col < 0 || col >= w) return 0;
    return im[(channel*h + row)*w + col];
}

void im2col_cpu(float *im, int h, int w, int c, int size, int stride, int pad, float *output)
{
    int i, j, k;
    int num = size * size * c;
    int out_h = (h + 2*pad - size) / stride + 1;
    int out_w = (w + 2*pad - size) / stride + 1;
    for (i = 0; i < num; ++i){
        int h_offset = (i / size) % size;
        int w_offset = i % size;
        int c_offset = i / size / size;
        for (j = 0; j < out_h; ++j){
            for (k = 0; k < out_w; ++k){
                int row = stride * j + h_offset;
                int col = stride * k + w_offset;
                int channel = c_offset;
                int index = (i*out_h + j)*out_w + k;
                output[index] = get_pixel(im, row, col, channel, h, w, c, pad);
            }
        }
    }
}