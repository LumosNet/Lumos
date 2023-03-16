#ifndef BENCHMARK_JSON_H
#define BENCHMARK_JSON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"

#ifdef GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

char *load_from_json_file(char *path);
void load_params(cJSON *cjson_benchmark, char **param_names, void **space, int num);
int load_param(cJSON *cjson_benchmark, char *param_name, void **space, int index);

void load_float_array(cJSON *cjson_value, void *space, int num);
void load_int_array(cJSON *cjson_value, void *space, int num);

#ifdef __cplusplus
}
#endif

#endif
