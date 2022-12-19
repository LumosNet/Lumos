#ifndef COMPARE_H
#define COMPARE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cJSON.h"
#include "cJSON_Utils.h"
#include "benchmark_json.h"

#ifdef GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define PASS    0
#define ERROR   1

int compare_test(cJSON *cjson_benchmark, void **origin, char **bnames, int num);
int compare_float_array(float *a, float *b, int num);
int compare_int_array(int *a, int *b, int num);

#ifdef GPU
int compare_test_gpu(cJSON *cjson_benchmark, void **origin, char **bnames, int num);
int compare_float_array_gpu(float *a, float *b, int num);
int compare_int_array_gpu(int *a, int *b, int num);
#endif

#ifdef __cplusplus
}
#endif

#endif
