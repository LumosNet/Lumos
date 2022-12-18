#include "tsession.h"

void run_benchmarks(char *benchmark)
{
    cJSON *cjson_benchmark = NULL;
    cJSON *cjson_public = NULL;
    cJSON *cjson_interface = NULL;
    cJSON *cjson_type = NULL;
    cJSON *cjson_benchmarks = NULL;
    cJSON *cjson_params = NULL;
    cJSON *cjson_compare_type = NULL;
    cJSON *cjson_benchmark_item = NULL;
    cJSON *cjson_param_item = NULL;
    cJSON *cjson_single_benchmark = NULL;
    char *interface = NULL;
    char *type = NULL;
    char **benchmarks = NULL;
    char **params = NULL;
    char *compare_type = NULL;
    int benchmark_num = 0;
    int param_num = 0;
    int bvalue_num = 0;
    void **space = NULL;
    void **ret = malloc(sizeof(void*));
    int all_pass = 0;
    char *json_benchmark = load_from_json_file(benchmark);
    cjson_benchmark = cJSON_Parse(json_benchmark);
    cjson_public = cJSON_GetObjectItem(cjson_benchmark, "Public");
    cjson_interface = cJSON_GetObjectItem(cjson_public, "interface");
    cjson_type = cJSON_GetObjectItem(cjson_public, "type");
    cjson_benchmarks = cJSON_GetObjectItem(cjson_public, "benchmarks");
    cjson_params = cJSON_GetObjectItem(cjson_public, "params");
    cjson_compare_type = cJSON_GetObjectItem(cjson_public, "compare_type");
    interface = cjson_interface->valuestring;
    type = cjson_type->valuestring;
    compare_type = cjson_compare_type->valuestring;
    benchmark_num = cJSON_GetArraySize(cjson_benchmarks);
    param_num = cJSON_GetArraySize(cjson_params);
    benchmarks = malloc(benchmark_num*sizeof(char*));
    params = malloc(param_num*sizeof(char*));
    space = malloc((param_num+1)*sizeof(void*));
    for (int i = 0; i < benchmark_num; ++i){
        cjson_benchmark_item = cJSON_GetArrayItem(cjson_benchmarks, i);
        benchmarks[i] = cjson_benchmark_item->valuestring;
    }
    for (int i = 0; i < param_num; ++i){
        cjson_param_item = cJSON_GetArrayItem(cjson_params, i);
        params[i] = cjson_param_item->valuestring;
    }
    test_run(interface);
    for (int i = 0; i < benchmark_num; ++i){
        int compare_flag = 1;
        cjson_single_benchmark = cJSON_GetObjectItem(cjson_benchmark, benchmarks[i]);
        load_params(cjson_single_benchmark, params, space, param_num);
        bvalue_num = load_param(cjson_single_benchmark, "benchmark", space, param_num);
        if (0 == strcmp(type, "ops")){
            call_ops(interface, space, ret);
        }
        if (0 == strcmp(compare_type, "float")){
            compare_flag = compare_float_array(space[param_num], (float*)ret[0], bvalue_num);
        }
        if (0 == strcmp(compare_type, "int")){
            compare_flag = compare_int_array(space[param_num], (int*)ret[0], bvalue_num);
        }
        if (compare_flag == 0){
            test_msg_pass(benchmarks[i]);
        }
        else{
            all_pass = 1;
            test_msg_error(benchmarks[i]);
        }
    }
    release_params_space(space, param_num+1);
    test_res(all_pass, " ");
}
