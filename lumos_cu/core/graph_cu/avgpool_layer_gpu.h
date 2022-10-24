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



#endif

#ifdef __cplusplus
}
#endif
#endif