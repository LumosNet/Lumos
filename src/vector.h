#ifndef Tensor_H
#define Tensor_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lumos.h"
#include "tensor.h"
#include "array.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef VECTOR
float norm1_vt(Tensor *ts);
float norm2_vt(Tensor *ts);
float normp_vt(Tensor *ts, int p);
float infnorm_vt(Tensor *ts);
float ninfnorm_vt(Tensor *ts);
#endif

#ifdef  __cplusplus
}
#endif

#endif