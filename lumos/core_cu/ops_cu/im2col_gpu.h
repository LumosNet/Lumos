#ifndef CPU_GPU_H
#define CPU_GPU_H

#ifdef GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU

void im2col_gpu(float *img, int height, int width, int channel, int ksize, int stride, int pad, float *space);
void col2im_gpu(float *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, float *space);

#endif

#ifdef __cplusplus
}
#endif
#endif
