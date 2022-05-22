#ifndef IMAGE_H
#define IMAGE_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "im2col.h"

#ifdef __cplusplus
extern "C"{
#endif

// 统计图像中不同灰度等级的像素点数量
int *census_image_pixel(float *img, int w, int h, int c);
// 统计通道中不同灰度等级的像素点数量
int *census_channel_pixel(float *img, int w, int h, int c, int index_c);

float *load_image_data(char *img_path, int *w, int *h, int *c);
void save_image_data(float *img, int w, int h, int c, char *savepath);

// 双线性内插值
void resize_im(float *img, int height, int width, int channel, int row, int col, float *space);

#ifdef  __cplusplus
}
#endif

#endif