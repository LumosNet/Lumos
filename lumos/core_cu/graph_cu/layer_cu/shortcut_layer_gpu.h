#ifndef SHORTCUT_LAYER_GPU_H
#define SHORTCUT_LAYER_GPU_H

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "layer.h"
#include "shortcut_gpu.h"
#include "cpu_gpu.h"
#include "active_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_shortcut_layer_gpu(Layer l, int num);
void backward_shortcut_layer_gpu(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif
#endif
