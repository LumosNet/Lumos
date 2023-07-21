#ifndef COMPARE_H
#define COMPARE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cJSON.h"
#include "cJSON_Utils.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define ERROR   0
#define PASS    1

int compare_array(void *a, void *b, char *type, int num);
int compare_float_array(float *a, float *b, int num);
int compare_int_array(int *a, int *b, int num);

int compare_array_gpu(void *a, void *b, char *type, int num);
int compare_float_array_gpu(float *a, float *b, int num);
int compare_int_array_gpu(int *a, int *b, int num);

#ifdef __cplusplus
}
#endif

#endif
