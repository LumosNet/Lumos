#include "analysis_benchmark_file.h"

cJSON *get_benchmark(char *path)
{
    cJSON *benchmark = NULL;
    FILE *fp = fopen(path, "r");
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    benchmark = cJSON_Parse(tmp);
    return benchmark;
}

cJSON *get_public(cJSON *benchmark)
{
    cJSON *cjson_public = cJSON_GetObjectItem(benchmark, "Public");
    return cjson_public;
}

char *load_interface(cJSON *public)
{
    cJSON *cjson_interface = NULL;
    char *interface = NULL;
    cjson_interface = cJSON_GetObjectItem(public, "interface");
    interface = cjson_interface->valuestring;
    return interface;
}

int load_cases_name(cJSON *public, char **benchmarks)
{
    cJSON *cjson_benchmarks = NULL;
    cJSON *cjson_benchmark_item = NULL;
    int benchmark_num = 0;
    cjson_benchmarks = cJSON_GetObjectItem(public, "benchmarks");
    benchmark_num = cJSON_GetArraySize(cjson_benchmarks);
    benchmarks = malloc(benchmark_num*sizeof(char*));
    for (int i = 0; i < benchmark_num; ++i){
        cjson_benchmark_item = cJSON_GetArrayItem(cjson_benchmarks, i);
        benchmarks[i] = cjson_benchmark_item->valuestring;
    }
    return benchmark_num;
}

int load_params_name(cJSON *public, char **params)
{
    cJSON *cjson_params = NULL;
    cJSON *cjson_param_item = NULL;
    int param_num = 0;
    cjson_params = cJSON_GetObjectItem(public, "params");
    param_num = cJSON_GetArraySize(cjson_params);
    params = malloc(param_num*sizeof(char*));
    for (int i = 0; i < param_num; ++i){
        cjson_param_item = cJSON_GetArrayItem(cjson_params, i);
        params[i] = cjson_param_item->valuestring;
    }
    return param_num;
}

int load_compares_name(cJSON *public, char **compares)
{
    cJSON *cjson_compares = NULL;
    cJSON *cjson_compare_item = NULL;
    int compare_num = 0;
    cjson_compares = cJSON_GetObjectItem(public, "compares");
    compare_num = cJSON_GetArraySize(cjson_compares);
    compares = malloc(compare_num*sizeof(char*));
    for (int i = 0; i < compare_num; ++i){
        cjson_compare_item = cJSON_GetArrayItem(cjson_compares, i);
        compares[i] = cjson_compare_item->valuestring;
    }
    return compare_num;
}

void get_params_value(cJSON *single_benchmark, char **params, int param_num, void **space, int *num_list, char **types)
{
    load_params(single_benchmark, params, space, num_list, types, param_num);
}

void get_compare_value(cJSON *single_benchmark, char **compares, int compare_num, void **space, int *num_list, char **types)
{
    cJSON *cjson_benchmark_value = cJSON_GetObjectItem(single_benchmark, "benchmark");
    load_params(cjson_benchmark_value, compares, space, num_list, types, compare_num);
}

void get_params_value_gpu(cJSON *single_benchmark, char **params, int param_num, void **space, int *num_list, char **types)
{
    load_params_gpu(single_benchmark, params, space, num_list, types, param_num);
}

void get_compare_value_gpu(cJSON *single_benchmark, char **compares, int compare_num, void **space, int *num_list, char **types)
{
    cJSON *cjson_benchmark_value = cJSON_GetObjectItem(single_benchmark, "benchmark");
    load_params_gpu(cjson_benchmark_value, compares, space, num_list, types, compare_num);
}

void load_params(cJSON *cjson_benchmark, char **param_names, void **space, int *num_list, char **types, int num)
{
    for (int i = 0; i < num; ++i){
        num_list[i] = load_param(cjson_benchmark, param_names[i], space, types, i);
    }
}

int load_param(cJSON *cjson_benchmark, char *param_name, void **space, char **types, int index)
{
    int size = 0;
    void *value = NULL;
    cJSON *cjson_param = NULL;
    cJSON *cjson_type = NULL;
    cJSON *cjson_value = NULL;
    cjson_param = cJSON_GetObjectItem(cjson_benchmark, param_name);
    cjson_type = cJSON_GetObjectItem(cjson_param, "type");
    cjson_value = cJSON_GetObjectItem(cjson_param, "value");
    char *type = cjson_type->valuestring;
    if (0 == strcmp(type, "string")){
        value = cJSON_GetStringValue(cjson_value);
        size = 1;
    } else {
        size = cJSON_GetArraySize(cjson_value);
        if (0 == strcmp(type, "float")){
            value = (void*)malloc(size*sizeof(float));
            load_float_array(cjson_value, value, size);
        } else if (0 == strcmp(type, "int")){
            value = (void*)malloc(size*sizeof(int));
            load_int_array(cjson_value, value, size);
        }
    }
    space[index] = value;
    types[index] = type;
    return size;
}

void load_params_gpu(cJSON *cjson_benchmark, char **param_names, void **space, int *num_list, char **types, int num)
{
    for (int i = 0; i < num; ++i){
        num_list[i] = load_param_gpu(cjson_benchmark, param_names[i], space, types, i);
    }
}

int load_param_gpu(cJSON *cjson_benchmark, char *param_name, void **space, char **types, int index)
{
    int size = 0;
    void *value = NULL;
    void *value_gpu = NULL;
    cJSON *cjson_param = NULL;
    cJSON *cjson_type = NULL;
    cJSON *cjson_value = NULL;
    cjson_param = cJSON_GetObjectItem(cjson_benchmark, param_name);
    cjson_type = cJSON_GetObjectItem(cjson_param, "type");
    cjson_value = cJSON_GetObjectItem(cjson_param, "value");
    char *type = cjson_type->valuestring;
    if (0 == strcmp(type, "string")){
        value = cJSON_GetStringValue(cjson_value);
        size = 1;
    }
    else {
        size = cJSON_GetArraySize(cjson_value);
        if (0 == strcmp(type, "float")){
            value = (void*)malloc(size*sizeof(float));
            cudaMalloc((void**)&value_gpu, size*sizeof(float));
            load_float_array(cjson_value, value, size);
            cudaMemcpy(value_gpu, value, size*sizeof(float), cudaMemcpyHostToDevice);
        } else if (0 == strcmp(type, "int")){
            value = (void*)malloc(size*sizeof(int));
            cudaMalloc((void**)&value_gpu, size*sizeof(int));
            load_int_array(cjson_value, value, size);
            cudaMemcpy(value_gpu, value, size*sizeof(int), cudaMemcpyHostToDevice);
        }
    }
    space[index] = value_gpu;
    types[index] = type;
    free(value);
    return size;
}

void load_float_array(cJSON *cjson_value, void *space, int num)
{
    float *values = (float*)space;
    cJSON *cjson_array_item = NULL;
    for (int i = 0; i < num; ++i){
        cjson_array_item = cJSON_GetArrayItem(cjson_value, i);
        values[i] = cjson_array_item->valuedouble;
    }
}

void load_int_array(cJSON *cjson_value, void *space, int num)
{
    int *values = (int*)space;
    cJSON *cjson_array_item = NULL;
    for (int i = 0; i < num; ++i){
        cjson_array_item = cJSON_GetArrayItem(cjson_value, i);
        values[i] = cjson_array_item->valueint;
    }
}
