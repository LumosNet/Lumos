#ifndef GRAY_PROCESS_H
#define GRAY_PROCESS_H

#include <math.h>

#include "tensor.h"

// 图像反转
Image *img_reversal(Image *img);

// 对数变换
Image *Log_transfer(Image *img, float c);

// 幂律（伽马）变换
Image *power_law(Image *img, float c, float gamma, float varepsilon);

// 对比度拉伸
Image *contrast_stretch(Image *img, float r_max, float r_min, float s_max, float s_min);

// 灰度分层（slice作为划分像素值，大于等于该像素值的像素用focus填充，小于该像素值的像素用no_focus填充）
Image *gray_slice_1(Image *img, float slice, float focus, float no_focus);
// 灰度分层（slice作为划分像素值，flag=1,大于等于slice的像素值用focus填充，其余像素值不变，反之亦然）
Image *gray_slice_2(Image *img, float slice, int flag, float focus);

// 8比特分层
Image **bit8_slice(Image *img);
// 比特重构
Image *bit_restructure(Image **imgs, int *bit_index, int len);

// 直方图均衡
Image *histogram_equalization(Image *img);
// 对图像某一channel进行直方图均衡
void __histogram_equalization_channel(Image *o_img, Image *a_img, int c);

// 直方图匹配/直方图规定化

#endif