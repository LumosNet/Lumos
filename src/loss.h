#ifndef LOSS_H
#define LOSS_H

#include <math.h>

#include "tensor.h"
#include "array.h"
#include "Vector.h"

#include "umath.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef float (*LossFunc)();

float mse(Vector *yi, Vector *yh);
float mae(Vector *yi, Vector *yh);

float huber(Vector *yi, Vector *yh, float theta);
float quantile(Vector *yi, Vector *yh, float r);

float cross_entropy(Vector *yi, Vector *yh);
float hinge(Vector *yi, Vector *yh);

#ifdef  __cplusplus
}
#endif

#endif