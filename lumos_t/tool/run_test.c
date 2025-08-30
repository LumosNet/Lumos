#include "run_test.h"

int run_by_benchmark_file(char *path, TestInterface FUNC, int coretype, FILE *logfp)
{
    if (0 == strcmp(path, "NULL")) return -1;
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
            get_copy_value_cpu(space, compare, params, params_num_list, compares_num_list, params_types, compares_types, compares_num, params_num);
            fprintf(stderr, "  Load running params\n");
            FUNC(space, ret);
        } else {
            get_params_value_gpu(CJsinglebench, params, params_num, space, params_num_list, params_types);
            get_compare_value_gpu(CJsinglebench, compares, compares_num, compare, compares_num_list, compares_types);
            get_copy_value_gpu(space, compare, params, params_num_list, compares_num_list, params_types, compares_types, compares_num, params_num);
            fprintf(stderr, "  Load running params\n");
            FUNC(space, ret);
        }
        fprintf(stderr, "  Running test case \e[0;32mFINISH\e[0m\n");
        for (int j = 0; j < compares_num; ++j){
            if (coretype == CPU){
                flag = compare_array(compare[j], ret[j], compares_types[j], compares_num_list[j], logfp);
            } else {
                flag = compare_array_gpu(compare[j], ret[j], compares_types[j], compares_num_list[j], logfp);
            }
            if (flag == 1){
                fprintf(stderr, "  Interface %s: %s \e[0;32mPASS\e[0m\n", interface, compares[j]);
            } else {
                logging_msg(3, "Compare Data:\n", logfp);
                logging_data(compares_types[j], compare[j], 1, compares_num_list[j], 1, logfp);
                logging_msg(3, "Return Data:\n", logfp);
                logging_data(compares_types[j], ret[j], 1, compares_num_list[j], 1, logfp);
                fprintf(stderr, "  Interface %s: %s \e[0;31mFAIL\e[0m\n", interface, compares[j]);
                all_flag = 0;
            }
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

int run_all_benchmark(int coretype, FILE *logfp)
{
    char *interface_list = "./lumos_t/benchmark/all";
    char *tmp = fget(interface_list);
    int *index = split(tmp, '\n');
    int lines = index[0];
    char *interface_line = NULL;
    int *interface_index = NULL;
    TestInterface FUNC = NULL;
    for (int i = 0; i < lines; ++i){
        interface_line = tmp+index[i+1];
        interface_index = split(interface_line, ' ');
        if (coretype == CPU) FUNC = get_interface_cpu(interface_line+interface_index[1]);
        else if (coretype == GPU) FUNC = get_interface_gpu(interface_line+interface_index[1]);;
        run_by_benchmark_file(interface_line+interface_index[2], FUNC, coretype, logfp);
    }
    return 0;
}

TestInterface get_interface_cpu(char *name)
{
    if (0 == strcmp(name, "add_bias")) return call_add_bias;
    else if (0 == strcmp(name, "scale_bias")) return call_scale_bias;
    else if (0 == strcmp(name, "fill_cpu")) return call_fill_cpu;
    else if (0 == strcmp(name, "multy_cpu")) return call_multy_cpu;
    else if (0 == strcmp(name, "add_cpu")) return call_add_cpu;
    else if (0 == strcmp(name, "min_cpu")) return call_min_cpu;
    else if (0 == strcmp(name, "max_cpu")) return call_max_cpu;
    else if (0 == strcmp(name, "sum_cpu")) return call_sum_cpu;
    else if (0 == strcmp(name, "mean_cpu")) return call_mean_cpu;
    else if (0 == strcmp(name, "matrix_add_cpu")) return call_matrix_add_cpu;
    else if (0 == strcmp(name, "matrix_subtract_cpu")) return call_matrix_subtract_cpu;
    else if (0 == strcmp(name, "matrix_multiply_cpu")) return call_matrix_multiply_cpu;
    else if (0 == strcmp(name, "matrix_divide_cpu")) return call_matrix_divide_cpu;
    else if (0 == strcmp(name, "saxpy_cpu")) return call_saxpy_cpu;
    else if (0 == strcmp(name, "sum_channel_cpu")) return call_sum_channel_cpu;
    else if (0 == strcmp(name, "one_hot_encoding")) return call_one_hot_encoding;
    else if (0 == strcmp(name, "gemm")) return call_gemm;
    else if (0 == strcmp(name, "gemm_nn")) return call_gemm_nn;
    else if (0 == strcmp(name, "gemm_tn")) return call_gemm_tn;
    else if (0 == strcmp(name, "gemm_nt")) return call_gemm_nt;
    else if (0 == strcmp(name, "gemm_tt")) return call_gemm_tt;
    else if (0 == strcmp(name, "im2col")) return call_im2col;
    else if (0 == strcmp(name, "col2im")) return call_col2im;
    else if (0 == strcmp(name, "census_image_pixel")) return call_census_image_pixel;
    else if (0 == strcmp(name, "census_channel_pixel")) return call_census_channel_pixel;
    else if (0 == strcmp(name, "load_image_data")) return call_load_image_data;
    else if (0 == strcmp(name, "save_image_data")) return call_save_image_data;
    else if (0 == strcmp(name, "resize_im")) return call_resize_im;
    else if (0 == strcmp(name, "avgpool")) return call_avgpool;
    else if (0 == strcmp(name, "maxpool")) return call_maxpool;
    else if (0 == strcmp(name, "avgpool_gradient")) return call_avgpool_gradient;
    else if (0 == strcmp(name, "maxpool_gradient")) return call_maxpool_gradient;
    else return NULL;
}

TestInterface get_interface_gpu(char *name)
{
    if (0 == strcmp(name, "add_bias")) return call_add_bias_gpu;
    else if (0 == strcmp(name, "scale_bias")) return call_scale_bias_gpu;
    else if (0 == strcmp(name, "fill_cpu")) return call_fill_gpu;
    else if (0 == strcmp(name, "multy_cpu")) return call_multy_gpu;
    else if (0 == strcmp(name, "add_cpu")) return call_add_gpu;
    else if (0 == strcmp(name, "min_cpu")) return call_min_gpu;
    else if (0 == strcmp(name, "max_cpu")) return call_max_gpu;
    else if (0 == strcmp(name, "sum_cpu")) return call_sum_gpu;
    else if (0 == strcmp(name, "mean_cpu")) return call_mean_gpu;
    else if (0 == strcmp(name, "matrix_add_cpu")) return call_matrix_add_gpu;
    else if (0 == strcmp(name, "matrix_subtract_cpu")) return call_matrix_subtract_gpu;
    else if (0 == strcmp(name, "matrix_multiply_cpu")) return call_matrix_multiply_gpu;
    else if (0 == strcmp(name, "matrix_divide_cpu")) return call_matrix_divide_gpu;
    else if (0 == strcmp(name, "saxpy_cpu")) return call_saxpy_gpu;
    else if (0 == strcmp(name, "sum_channel_cpu")) return call_sum_channel_gpu;
    else if (0 == strcmp(name, "gemm")) return call_gemm_gpu;
    else if (0 == strcmp(name, "gemm_nn")) return call_gemm_nn_gpu;
    else if (0 == strcmp(name, "gemm_tn")) return call_gemm_tn_gpu;
    else if (0 == strcmp(name, "gemm_nt")) return call_gemm_nt_gpu;
    else if (0 == strcmp(name, "gemm_tt")) return call_gemm_tt_gpu;
    else if (0 == strcmp(name, "im2col")) return call_im2col_gpu;
    else if (0 == strcmp(name, "col2im")) return call_col2im_gpu;
    else if (0 == strcmp(name, "avgpool")) return call_avgpool_gpu;
    else if (0 == strcmp(name, "maxpool")) return call_maxpool_gpu;
    else if (0 == strcmp(name, "avgpool_gradient")) return call_avgpool_gradient_gpu;
    else if (0 == strcmp(name, "maxpool_gradient")) return call_maxpool_gradient_gpu;
    else return NULL;
}
