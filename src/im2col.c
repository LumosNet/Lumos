#include "im2col.h"

float im2col_get_pixel(float *img, int h, int w, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= h || col >= w) return 0;
    return img[col + w*(row + h*channel)];
}

void im2col(float *img, int h, int w, int c, int ksize, int stride, int pad, float *space)
{
    int height_col = (h + 2*pad - ksize) / stride + 1;
    int width_col = (w + 2*pad - ksize) / stride + 1;

    int channels = c * ksize * ksize;
    for (int c = 0; c < channels; ++c){
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_offset = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h){
            for (int w = 0; w < width_col; ++w){
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (h*width_col + w)*channels + c;
                space[col_index] = im2col_get_pixel(img, h, w, im_col, im_row, c_offset, pad);
            }
        }
    }
}

void col2im(float *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, float *space)
{
    for (int c = 0; c < out_c; ++c){
        for (int i = 0; i < out_h; ++i){
            int kernel_h_index = (i+pad) / stride;
            int h_index = (i+pad) % stride;
            int o_flag = -1;
            if (kernel_h_index+1 > out_h) o_flag = 1;
            else if (h_index+1 > ksize) o_flag = 1;
            for (int j = 0; j < out_w; ++j){
                if (o_flag) space[c*out_h*out_w + i*out_w + j] = 0;
                else {
                    int kernel_w_index = (j+pad) / stride;
                    int w_index = (j+pad) % stride;
                    if (kernel_w_index+1 > out_w) space[c*out_h*out_w + i*out_w + j] = 0;
                    else if (w_index+1 > ksize) space[c*out_h*out_w + i*out_w + j] = 0;
                    else {
                        int index_h = kernel_h_index*out_w + kernel_w_index;
                        int index_w = c*ksize*ksize + h_index*ksize + w_index;
                        space[c*out_h*out_w + i*out_w + j] = img[index_h*ksize*ksize*out_c + index_w];
                    }
                }
            }
        }
    }
}