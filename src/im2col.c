#include "im2col.h"

float im2col_get_pixel(float *img, int h, int w, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= h || col >= w) return 0;
    return img[col + w*(row + h*channel)];
}

void im2col(float *img, int height, int width, int channel, int ksize, int stride, int pad, float *space)
{
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channel * ksize * ksize;
    for (int c = 0; c < channels_col; ++c){
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_offset = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h){
            for (int w = 0; w < width_col; ++w){
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (h*width_col + w)*channels_col + c;
                space[col_index] = im2col_get_pixel(img, height, width, im_row, im_col, c_offset, pad);
            }
        }
    }
}

void col2im(float *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, float *space)
{
    int height_col = (out_h + 2*pad - ksize) / stride + 1;
    int width_col = (out_w + 2*pad - ksize) / stride + 1;
    for (int c = 0; c < out_c; ++c){
        for (int i = 0; i < out_h; ++i){
            int kernel_h_index = (i+pad) / stride;
            int h_index = (i+pad) % stride;
            int o_flag = 0;
            if (kernel_h_index+1 > height_col){
                h_index = (kernel_h_index - height_col + 1) * stride + h_index;
                kernel_h_index = height_col - 1;
            }
            if (h_index+1 > ksize) o_flag = 1;
            for (int j = 0; j < out_w; ++j){
                if (o_flag) space[c*out_h*out_w + i*out_w + j] = 0;
                else {
                    int kernel_w_index = (j+pad) / stride;
                    int w_index = (j+pad) % stride;
                    if (kernel_w_index+1 > width_col){
                        w_index = (kernel_w_index - width_col + 1) * stride + w_index;
                        kernel_w_index = width_col - 1;
                    }
                    if (w_index+1 > ksize) space[c*out_h*out_w + i*out_w + j] = 0;
                    else {
                        int index_h = kernel_h_index*width_col + kernel_w_index;
                        int index_w = c*ksize*ksize + h_index*ksize + w_index;
                        space[c*out_h*out_w + i*out_w + j] = img[index_h*ksize*ksize*out_c + index_w];
                    }
                }
            }
        }
    }
}