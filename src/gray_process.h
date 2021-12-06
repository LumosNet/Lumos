#ifndef GRAY_PROCESS_H
#define GRAY_PROCESS_H

#include <math.h>

#include "image.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GRAY
// 图像反转
Tensor *img_reversal(Tensor *img);

// 对数变换
Tensor *Log_transfer(Tensor *img, float c);

// 幂律（伽马）变换
Tensor *power_law(Tensor *img, float c, float gamma, float varepsilon);

// 对比度拉伸
Tensor *contrast_stretch(Tensor *img, float r_max, float r_min, float s_max, float s_min);

// 灰度分层（slice作为划分像素值，大于等于该像素值的像素用focus填充，小于该像素值的像素用no_focus填充）
Tensor *gray_slice_1(Tensor *img, float slice, float focus, float no_focus);
// 灰度分层（slice作为划分像素值，flag=1,大于等于slice的像素值用focus填充，其余像素值不变，反之亦然）
Tensor *gray_slice_2(Tensor *img, float slice, int flag, float focus);

// 8比特分层
Tensor **bit8_slice(Tensor *img);
// 比特重构
Tensor *bit_restructure(Tensor **imgs, int *bit_index, int len);

// 直方图均衡
Tensor *histogram_equalization(Tensor *img);
// 对图像某一channel进行直方图均衡
void __histogram_equalization_channel(Tensor *o_img, Tensor *a_img, int c);
#endif

#ifdef  __cplusplus
}
#endif

#endif