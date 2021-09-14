#ifndef IMAGE_H
#define IMAGE_H

#include <string.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C"{
#endif

typedef Tensor Image;

Image *create_image(int w, int h, int c);

// 统计图像中不同灰度等级的像素点数量
int *census_image_pixel(Image *img);
// 统计通道中不同灰度等级的像素点数量
int *census_channel_pixel(Image *img, int c);

Image *load_image_data(char *img_path);
void save_image_data(Image *img, char *savepath);

#ifdef  __cplusplus
}
#endif

#endif