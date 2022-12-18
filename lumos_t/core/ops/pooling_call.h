#ifndef POOLING_CALL_H
#define POOLING_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pooling.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_avgpool(void **params, void **ret);
void call_maxpool(void **params, void **ret);

void call_avgpool_gradient(void **params, void **ret);
void call_maxpool_gradient(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
