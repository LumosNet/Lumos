#ifndef POOLING_GPU_H
#define POOLING_GPU_H

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

void avgpool_gpu(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space);
void maxpool_gpu(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index);

void avgpool_gradient_gpu(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n);
void maxpool_gradient_gpu(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n, int *index);

#endif

#ifdef __cplusplus
}
#endif
#endif
