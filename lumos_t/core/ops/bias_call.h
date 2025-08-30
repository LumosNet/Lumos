#ifndef BIAS_CALL_H
#define BIAS_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bias.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_add_bias(void **params, void **ret);
void call_scale_bias(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
