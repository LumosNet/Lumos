#ifndef IMAGE_H
#define IMAGE_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "im2col.h"

#ifdef __cplusplus
extern "C"{
#endif

typedef Tensor Image;
typedef Tensor Kernel;

Image *create_image(int w, int h, int c);

// 统计图像中不同灰度等级的像素点数量
int *census_image_pixel(Image *img);
// 统计通道中不同灰度等级的像素点数量
int *census_channel_pixel(Image *img, int c);
// 获取图像通道数
int get_channels(Image *img);

Image *load_image_data(char *img_path);
void save_image_data(Image *img, char *savepath);

// 双线性内插值
Image *resize_ts_im(Image *img, int width, int height);

#ifdef  __cplusplus
}
#endif

#endif