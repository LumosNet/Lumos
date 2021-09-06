#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
#include "array.h"
#include "victor.h"

#ifdef  __cplusplus
extern "C" {
#endif

float mse(Victor *yi, Victor *yh);

#ifdef  __cplusplus
}
#endif

#endif