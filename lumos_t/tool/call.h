#ifndef CALL_H
#define CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bias_call.h"

#ifdef GPU
#include "bias_gpu_call.h"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

void call_ops(char *interface, void **params, void **ret);

#ifdef GPU
void call_cu_ops(char *interface, void **params, void **ret);
#endif

#ifdef __cplusplus
}
#endif

#endif
