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

Tensor *create_image(int w, int h, int c);

// 统计图像中不同灰度等级的像素点数量
int *census_image_pixel(Tensor *img);
// 统计通道中不同灰度等级的像素点数量
int *census_channel_pixel(Tensor *img, int c);
// 获取图像通道数
int get_channels(Tensor *img);

Tensor *load_image_data(char *img_path);
void save_image_data(Tensor *img, char *savepath);

// 双线性内插值
Tensor *resize_ts_im(Tensor *img, int width, int height);

#ifdef  __cplusplus
}
#endif

#endif