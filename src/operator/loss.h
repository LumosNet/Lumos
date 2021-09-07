#ifndef LOSS_H
#define LOSS_H

#include <math.h>

#include "tensor.h"
#include "array.h"
#include "victor.h"

#include "umath.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef float (*LossFunc)();

float mse(Victor *yi, Victor *yh);
float mae(Victor *yi, Victor *yh);

float huber(Victor *yi, Victor *yh, float theta);
float quantile(Victor *yi, Victor *yh, float r);

float cross_entropy(Victor *yi, Victor *yh);
float hinge(Victor *yi, Victor *yh);

#ifdef  __cplusplus
}
#endif

#endif