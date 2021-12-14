#ifndef LOSS_H
#define LOSS_H

#include <math.h>

#include "tensor.h"
#include "array.h"
#include "vector.h"

#include "umath.h"

#ifdef  __cplusplus
extern "C" {
#endif

int *one_hot_encoding(int n, int label);

typedef float (*LossFunc)();

/*
    yi为原标签
    yh为计算标签
*/

// 均方误差
float mse(Tensor *yi, Tensor *yh);
// 平均绝对值误差
float mae(Tensor *yi, Tensor *yh);
// 平滑平均绝对误差
float huber(Tensor *yi, Tensor *yh, float theta);
// 分位数损失
float quantile(Tensor *yi, Tensor *yh, float gamma);
// 交叉熵损失
float cross_entropy(Tensor *yi, Tensor *yh);

float hinge(Tensor *yi, Tensor *yh);

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