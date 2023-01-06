#ifndef IM2COL_CALL_H
#define IM2COL_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "im2col.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_im2col(void **params, void **ret);
void call_col2im(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
