#ifndef VARIANCE_GPU_CALL_H
#define VARIANCE_GPU_CALL_H

#include "session.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_variance_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
