#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void softmax(float *data, int num, float *space);
void softmax_grident(float *data, int num, float *space);

#ifdef __cplusplus
}
#endif

#endif