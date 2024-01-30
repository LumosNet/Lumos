#ifndef SHORTCUT_GPU_H
#define SHORTCUT_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void shortcut_gpu(float *add, int aw, int ah, int ac, float *out, int ow, int oh, int oc, float beta, float alpha, float *space);

#ifdef __cplusplus
}
#endif
#endif
