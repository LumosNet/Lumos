#ifndef GPU_H
#define GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "gpu.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCK 512

void check_error(cudaError_t status);
dim3 cuda_gridsize(size_t n);
void error(const char *s);

#ifdef __cplusplus
}
#endif
#endif
