#include "im2col.h"

Tensor *im2col(Image *img, int ksize, int stride, int pad)
{
    int channels = 0;
    if (img->dim == 2) channels = 1;
    else if (img->dim) channels = img->size[2];
    else return NULL;
    int height_col = (img->size[1] + 2*pad - ksize) / stride + 1;
    int width_col = (img->size[0] + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    float *data_col = calloc(height_col * width_col * ksize * ksize, sizeof(float));
    for (int c = 0; c < channels_col; ++c){
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_offset = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h){
            for (int w = 0; w < width_col; ++w){
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (h*width_col + w)*(ksize * ksize) + h_offset * ksize + w_offset;
                int index[] = {im_col+1-pad, im_row+1-pad, c_offset+1};
                float val = get_pixel(img, index);
                data_col[col_index] += val;
            }
        }
    }
    Array *colim = array_list(height_col*width_col, ksize*ksize, data_col);
    return colim;
}