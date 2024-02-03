#ifndef UPDATE_WEIGHTS_GPU_CALL_H
#define UPDATE_WEIGHTS_GPU_CALL_H

#include "session.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_update_weights_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
