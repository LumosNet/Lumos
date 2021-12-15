#ifndef LOSS_H
#define LOSS_H

#include <math.h>

#include "tensor.h"
#include "array.h"
#include "vector.h"
#include "gemm.h"
#include "umath.h"

#ifdef  __cplusplus
extern "C" {
#endif

float *one_hot_encoding(int n, int label);

/*
    yi为原标签
    yh为计算标签
*/

// 均方误差
float mse(float *yi, float *yh, int n, float *space);
// 平均绝对值误差
float mae(float *yi, float *yh, int n);
// 平滑平均绝对误差
float huber(float *yi, float *yh, int n, float theta);
// 分位数损失
float quantile(float *yi, float *yh, int n, float gamma);
// 交叉熵损失
float cross_entropy(float *yi, float *yh, int n);

float hinge(float *yi, float *yh, int n);

void forward_mse_loss(Layer l, Network net);
void forward_mae_loss(Layer l, Network net);
void forward_huber_loss(Layer l, Network net);
void forward_quantile_loss(Layer l, Network net);
void forward_cross_entropy_loss(Layer l, Network net);
void forward_hinge_loss(Layer l, Network net);

void backward_mse_loss(Layer l, Network net);
void backward_mae_loss(Layer l, Network net);
void backward_huber_loss(Layer l, Network net);
void backward_quantile_loss(Layer l, Network net);
void backward_cross_entropy_loss(Layer l, Network net);
void backward_hinge_loss(Layer l, Network net);

LossType load_loss_type(char *loss);

Forward load_forward_loss(LossType TYPE);
Backward load_backward_loss(LossType TYPE);

#ifdef  __cplusplus
}
#endif

#endif