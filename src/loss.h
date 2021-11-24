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
float mse(Vector *yi, Vector *yh);
// 平均绝对值误差
float mae(Vector *yi, Vector *yh);

float huber(Vector *yi, Vector *yh, float theta);
// 分位数损失
float quantile(Vector *yi, Vector *yh, float r);

// 交叉熵损失
float cross_entropy(Vector *yi, Vector *yh);

float hinge(Vector *yi, Vector *yh);

#ifdef  __cplusplus
}
#endif

#endif