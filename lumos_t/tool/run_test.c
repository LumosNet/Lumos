#include "run_test.h"

void run_by_benchmark_file(char *path, int coretype)
{
    cJSON *CJbenchmark = NULL;
    cJSON *CJpublic = NULL;
    cJSON *CJsinglebench = NULL;
    void **space = NULL;
    void **ret = NULL;
    void **compare = NULL;
    char *interface = NULL;
    char **cases = NULL;
    char **params = NULL;
    char **compares = NULL;

    int *params_num_list = NULL;
    int *compares_num_list = NULL;
    char **params_types = NULL;
    char **compares_types = NULL;

    int cases_num = 0;
    int params_num = 0;
    int compares_num = 0;

    int flag = 0;

    CJbenchmark = get_benchmark(path);
    CJpublic = cJSON_GetObjectItem(CJbenchmark, "Public");
    interface = load_interface(CJpublic);
    cases = load_cases_name(CJpublic, &cases_num);
    params = load_params_name(CJpublic, &params_num);
    compares = load_compares_name(CJpublic, &compares_num);

    space = malloc(params_num*sizeof(void*));
    ret = malloc(compares_num*sizeof(void*));
    compare = malloc(compares_num*sizeof(void*));
    params_num_list = malloc(params_num*sizeof(int));
    compares_num_list = malloc(compares_num*sizeof(int));
    params_types = malloc(params_num*sizeof(char*));
    compares_types = malloc(compares_num*sizeof(char*));
    for (int i = 0; i < cases_num; ++i){
        CJsinglebench = cJSON_GetObjectItem(CJbenchmark, cases[i]);
        if (coretype == CPU){
            get_params_value(CJsinglebench, params, params_num, space, params_num_list, params_types);
            get_compare_value(CJsinglebench, compares, compares_num, compare, compares_num_list, compares_types);
            flag = call(interface, space, ret);
        } else {
            get_params_value_gpu(CJsinglebench, params, params_num, space, params_num_list, params_types);
            get_compare_value_gpu(CJsinglebench, compares, compares_num, compare, compares_num_list, compares_types);
            flag = call_cu(interface, space, ret);
        }
        if (flag){
            for (int j = 0; j < compares_num; ++j){
                if (coretype == CPU){
                    flag = compare_array(compare[j], ret[j], compares_types[j], compares_num_list[j]);
                } else {
                    flag = compare_array_gpu(compare[j], ret[j], compares_types[j], compares_num_list[j]);
                }
            }
        } else {
            test_msg_error("Interface can't find, Please checkout your testlist");
            continue;
        }
    }
    test_res(flag, " ");
    for (int i = 0; i < params_num; ++i){
        if (coretype == CPU){
            free(space[i]);
        } else {
            cudaFree(space[i]);
        }
    }
    for (int i = 0; i < compares_num; ++i){
        if (coretype == CPU){
            free(compare[i]);
        } else {
            cudaFree(compare[i]);
        }
    }
    free(space);
    free(compare);
    free(ret);
    free(cases);
    free(params);
    free(compares);
    free(params_num_list);
    free(compares_num_list);
    free(params_types);
    free(compares_types);
}
