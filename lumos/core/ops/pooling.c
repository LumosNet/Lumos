#include "pooling.h"

void avgpool(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    for (int k = 0; k < c; ++k){
        for (int i = 0; i < out_h; ++i){
            for (int j = 0; j < out_w; ++j){
                int x = i*stride;
                int y = j*stride;
                float temp = 0;
                for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
                    for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
                        int index_i = x + ksize_i - pad;
                        int index_j = y + ksize_j - pad;
                        if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
                        temp += im[k*h*w + index_i*w + index_j];
                    }
                }
                space[k*out_h*out_w + i*out_w + j] = temp / (ksize*ksize);
            }
        }
    }
}

void avgpool_gradient(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    fill_cpu(delta_l, h*w*c, 0, 1);
    for (int k = 0; k < c; ++k){
        for (int i = 0; i < out_h; ++i){
            for (int j = 0; j < out_w; ++j){
                int x = i*stride;
                int y = j*stride;
                for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
                    for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
                        int index_i = x + ksize_i - pad;
                        int index_j = y + ksize_j - pad;
                        if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
                        delta_l[k*h*w + index_i*w + index_j] += delta_n[k*out_h*out_w + i*out_w + j] / (ksize*ksize);
                    }
                }
            }
        }
    }
}

void maxpool(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    for (int k = 0; k < c; ++k){
        for (int i = 0; i < out_h; ++i){
            for (int j = 0; j < out_w; ++j){
                int x = i*stride;
                int y = j*stride;
                int max_index = -1;
                float max = -99;
                for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
                    for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
                        int index_i = x + ksize_i - pad;
                        int index_j = y + ksize_j - pad;
                        if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
                        if (im[k*h*w + index_i*w + index_j] > max){
                            max = im[k*h*w + index_i*w + index_j];
                            max_index = k*h*w + index_i*w + index_j;
                        }
                    }
                }
                if (max_index == -1) max = 0;
                space[k*out_h*out_w + i*out_w + j] = max;
                index[k*out_h*out_w + i*out_w + j] = max_index;
            }
        }
    }
}

void maxpool_gradient(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n, int *index)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    fill_cpu(delta_l, h*w*c, 0, 1);
    for (int j = 0; j < out_h * out_w * c; ++j)
    {
        if (index[j] == -1) continue;
        delta_l[index[j]] += delta_n[j];
    }
}
