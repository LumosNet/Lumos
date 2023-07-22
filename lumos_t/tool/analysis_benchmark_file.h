#ifndef ANALYSIS_BENCHMARK_FILE
#define ANALYSIS_BENCHMARK_FILE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef  __cplusplus
extern "C" {
#endif

cJSON *get_benchmark(char *path);
cJSON *get_public(cJSON *benchmark);
char *load_interface(cJSON *public);
char **load_cases_name(cJSON *public, int *num);
char **load_params_name(cJSON *public, int *num);
char **load_compares_name(cJSON *public, int *num);

void get_params_value(cJSON *single_benchmark, char **params, int param_num, void **space, int *num_list, char **types);
void get_compare_value(cJSON *single_benchmark, char **compares, int compare_num, void **space, int *num_list, char **types);
void get_params_value_gpu(cJSON *single_benchmark, char **params, int param_num, void **space, int *num_list, char **types);
void get_compare_value_gpu(cJSON *single_benchmark, char **compares, int compare_num, void **space, int *num_list, char **types);

void load_params_gpu(cJSON *cjson_benchmark, char **param_names, void **space, int *num_list, char **types, int num);
int load_param_gpu(cJSON *cjson_benchmark, char *param_name, void **space, char **types, int index);

void load_params(cJSON *cjson_benchmark, char **param_names, void **space, int *num_list, char **types, int num);
int load_param(cJSON *cjson_benchmark, char *param_name, void **space, char **types, int index);

void load_float_array(cJSON *cjson_value, void *space, int num);
void load_int_array(cJSON *cjson_value, void *space, int num);

#ifdef __cplusplus
}
#endif

#endif
