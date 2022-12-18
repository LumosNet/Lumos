#ifndef CALL_H
#define CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bias_call.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_ops(char *interface, void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
