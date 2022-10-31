#ifndef AVGPOOL_LAYER_GPU_H
#define AVGPOOL_LAYER_GPU_H

#ifdef GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU

void forward_avgpool_layer_gpu(Layer l, int num);

#endif

#ifdef __cplusplus
}
#endif
#endif