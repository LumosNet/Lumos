#ifndef DROPOUT_RAND_GPU_CALL_H
#define DROPOUT_RAND_GPU_CALL_H

#include "session.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_dropout_rand_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
