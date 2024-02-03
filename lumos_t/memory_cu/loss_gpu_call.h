#ifndef LOSS_GPU_CALL_H
#define LOSS_GPU_CALL_H

#include "session.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_loss_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
