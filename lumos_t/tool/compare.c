#include "compare.h"

int compare_test(cJSON *cjson_benchmark, void **origin, char **bnames, int num)
{
    cJSON *cjson_bench = NULL;
    cJSON *cjson_type = NULL;
    char *type = NULL;
    void **value = malloc(sizeof(void*));
    int value_num = 0;
    int compare_flag = ERROR;
    int ret_flag = PASS;
    for (int i = 0; i < num; ++i){
        cjson_bench = cJSON_GetObjectItem(cjson_benchmark, bnames[i]);
        cjson_type = cJSON_GetObjectItem(cjson_bench, "type");
        type = cjson_type->valuestring;
        value_num = load_param(cjson_benchmark, bnames[i], value, 0);
        if (0 == strcmp(type, "float")){
            compare_flag = compare_float_array((float*)origin[i], (float*)value[0], value_num);
        } else{
            compare_flag = compare_int_array((int*)origin[i], (int*)value[0], value_num);
        }
        if (compare_flag == ERROR) ret_flag = ERROR;
    }
    return ret_flag;
}

int compare_float_array(float *a, float *b, int num)
{
    for (int i = 0; i < num; ++i){
        if (fabs(a[i]-b[i]) > 1e-6){
            return ERROR;
        }
    }
    return PASS;
}

int compare_int_array(int *a, int *b, int num)
{
    for (int i = 0; i < num; ++i){
        if (fabs(a[i]-b[i]) > 1e-6){
            return ERROR;
        }
    }
    return PASS;
}

int compare_string_array(char **a, char **b, int num)
{
    for (int i = 0; i < num; ++i){
        if (0 != strcmp(a[i], b[i])){
            return ERROR;
        }
    }
    return PASS;
}
