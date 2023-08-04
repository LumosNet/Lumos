#ifndef X_NORM_GPU_CALL_H
#define X_NORM_GPU_CALL_H

#include "session.h"
#include "manager.h"
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_x_norm_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
