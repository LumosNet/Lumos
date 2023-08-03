#include "run_test.h"

int run_by_benchmark_file(char *path, int coretype)
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
    int all_flag = 1;

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
    test_run(interface, coretype);
    for (int i = 0; i < cases_num; ++i){
        fprintf(stderr, "  Do benchmark: %s\n", cases[i]);
        CJsinglebench = cJSON_GetObjectItem(CJbenchmark, cases[i]);
        if (coretype == CPU){
            get_params_value(CJsinglebench, params, params_num, space, params_num_list, params_types);
            get_compare_value(CJsinglebench, compares, compares_num, compare, compares_num_list, compares_types);
            fprintf(stderr, "  Load running params\n");
            flag = call(interface, space, ret);
        } else {
            get_params_value_gpu(CJsinglebench, params, params_num, space, params_num_list, params_types);
            get_compare_value_gpu(CJsinglebench, compares, compares_num, compare, compares_num_list, compares_types);
            fprintf(stderr, "  Load running params\n");
            flag = call_cu(interface, space, ret);
        }
        if (flag){
            fprintf(stderr, "  Running test case \e[0;32mFINISH\e[0m\n");
            for (int j = 0; j < compares_num; ++j){
                if (coretype == CPU){
                    flag = compare_array(compare[j], ret[j], compares_types[j], compares_num_list[j]);
                } else {
                    flag = compare_array_gpu(compare[j], ret[j], compares_types[j], compares_num_list[j]);
                }
                if (flag == 1){
                    fprintf(stderr, "  Interface %s: %s \e[0;32mPASS\e[0m\n", interface, compares[j]);
                } else {
                    fprintf(stderr, "  Interface %s: %s \e[0;31mFAIL\e[0m\n", interface, compares[j]);
                    all_flag = 0;
                }
            }
        } else {
            fprintf(stderr, "  Running test case \e[0;31mERROR\e[0m\n");
            test_msg_error("Interface can't find, Please checkout your testlist");
            all_flag = 0;
            break;
        }
    }
    test_res(all_flag, "All Cases run finish");
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
    return all_flag;
}

void run_all(char *listpath, int coretype)
{
    FILE *fp = fopen(listpath, "r");
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    int head = 0;
    for (int i = 0; i < file_size; ++i){
        if (tmp[i] == '\n'){
            tmp[i] = '\0';
            run_by_benchmark_file(tmp+head, coretype);
            head = i+1;
        }
    }
}

void run_all_cases(char *listpath, int flag)
{
    if (flag == 0) run_all(listpath, CPU);
    else if (flag == 1) run_all(listpath, GPU);
    else if (flag == 2){
        run_all(listpath, CPU);
        run_all(listpath, GPU);
    } else {
        fprintf(stderr, "Test Running Flag ERROR!\n");
    }
}

void run_by_interface(char *interface, int coretype)
{
    char *benchmark = interface_to_benchmark(interface);
    printf("%s\n", benchmark);
    run_by_benchmark_file(benchmark, coretype);
}
