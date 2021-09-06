#ifndef IMAGE_H
#define IMAGE_H

#include "matrix.h"

typedef Matrix Image;

Image *create_image(int w, int h, int c);
Image *copy_image(Image *img);

float min_pixel(Image *img);
float max_pixel(Image *img);
float pixel_mean(Image *img);
int pixel_num_image(Image *img, float x);

// 统计图像中不同灰度等级的像素点数量
int *census_image_pixel(Image *img);
// 统计通道中不同灰度等级的像素点数量
int *census_channel_pixel(Image *img, int c);

Image *load_image_data(char *img_path);
void save_image_data(Image *img, char *savepath);
#endif