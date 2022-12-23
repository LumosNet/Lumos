#include "benchmark_json.h"

char *load_from_json_file(char *path)
{
    FILE *fp = fopen(path, "r");
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    return tmp;
}

void load_params(cJSON *cjson_benchmark, char **param_names, void **space, int num)
{
    for (int i = 0; i < num; ++i){
        load_param(cjson_benchmark, param_names[i], space, i);
    }
}

int load_param(cJSON *cjson_benchmark, char *param_name, void **space, int index)
{
    void *value = NULL;
    cJSON *cjson_param = NULL;
    cJSON *cjson_type = NULL;
    cJSON *cjson_value = NULL;
    cjson_param = cJSON_GetObjectItem(cjson_benchmark, param_name);
    cjson_type = cJSON_GetObjectItem(cjson_param, "type");
    cjson_value = cJSON_GetObjectItem(cjson_param, "value");
    int size = cJSON_GetArraySize(cjson_value);
    char *type = cjson_type->valuestring;
    if (0 == strcmp(type, "float")){
        value = (void*)malloc(size*sizeof(float));
        load_float_array(cjson_value, value, size);
    } else if (0 == strcmp(type, "int")){
        value = (void*)malloc(size*sizeof(int));
        load_int_array(cjson_value, value, size);
    }
    space[index] = value;
    return size;
}

#ifdef GPU
void load_params_gpu(cJSON *cjson_benchmark, char **param_names, void **space, int num)
{
    for (int i = 0; i < num; ++i){
        load_param_gpu(cjson_benchmark, param_names[i], space, i);
    }
}

int load_param_gpu(cJSON *cjson_benchmark, char *param_name, void **space, int index)
{
    void *value = NULL;
    void *value_gpu = NULL;
    cJSON *cjson_param = NULL;
    cJSON *cjson_type = NULL;
    cJSON *cjson_value = NULL;
    cjson_param = cJSON_GetObjectItem(cjson_benchmark, param_name);
    cjson_type = cJSON_GetObjectItem(cjson_param, "type");
    cjson_value = cJSON_GetObjectItem(cjson_param, "value");
    int size = cJSON_GetArraySize(cjson_value);
    char *type = cjson_type->valuestring;
    if (0 == strcmp(type, "float")){
        value = (void*)malloc(size*sizeof(float));
        cudaMalloc((void**)&value_gpu, size*sizeof(float));
        load_float_array(cjson_value, value, size);
        cudaMemcpy(value, value_gpu, size*sizeof(float), cudaMemcpyHostToDevice);
    } else if (0 == strcmp(type, "int")){
        value = (void*)malloc(size*sizeof(int));
        cudaMalloc((void**)&value_gpu, size*sizeof(int));
        load_int_array(cjson_value, value, size);
        cudaMemcpy(value, value_gpu, size*sizeof(int), cudaMemcpyHostToDevice);
    }
    space[index] = value_gpu;
    free(value);
    return size;
}
#endif

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
