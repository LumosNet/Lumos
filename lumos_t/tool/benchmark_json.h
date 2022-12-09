#ifndef BENCHMARK_JSON_H
#define BENCHMARK_JSON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"

#ifdef  __cplusplus
extern "C" {
#endif

char *load_from_json_file(char *path);

void load_param(cJSON *cjson_benchmark, char *param_name, void **space, int index);
void load_float_array(cJSON *cjson_value, void *space, int num);
void load_int_array(cJSON *cjson_value, void *space, int num);

#ifdef __cplusplus
}
#endif

#endif
