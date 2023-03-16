#include "tsession.h"

void run_benchmarks(char *benchmark)
{
    cJSON *cjson_benchmark = NULL;
    cJSON *cjson_public = NULL;
    cJSON *cjson_interface = NULL;
    cJSON *cjson_type = NULL;
    cJSON *cjson_benchmarks = NULL;
    cJSON *cjson_params = NULL;
    cJSON *cjson_compares = NULL;
    cJSON *cjson_benchmark_item = NULL;
    cJSON *cjson_param_item = NULL;
    cJSON *cjson_compare_item = NULL;
    cJSON *cjson_single_benchmark = NULL;
    cJSON *cjson_benchmark_value = NULL;
    char *interface = NULL;
    char *type = NULL;
    char **benchmarks = NULL;
    char **params = NULL;
    char **compares = NULL;
    int benchmark_num = 0;
    int param_num = 0;
    int compare_num = 0;
    void **space = NULL;
    void **ret = NULL;
    int all_pass = 0;
    char *json_benchmark = load_from_json_file(benchmark);
    cjson_benchmark = cJSON_Parse(json_benchmark);
    cjson_public = cJSON_GetObjectItem(cjson_benchmark, "Public");
    cjson_interface = cJSON_GetObjectItem(cjson_public, "interface");
    cjson_type = cJSON_GetObjectItem(cjson_public, "type");
    cjson_benchmarks = cJSON_GetObjectItem(cjson_public, "benchmarks");
    cjson_params = cJSON_GetObjectItem(cjson_public, "params");
    cjson_compares = cJSON_GetObjectItem(cjson_public, "compares");
    interface = cjson_interface->valuestring;
    type = cjson_type->valuestring;
    benchmark_num = cJSON_GetArraySize(cjson_benchmarks);
    param_num = cJSON_GetArraySize(cjson_params);
    compare_num = cJSON_GetArraySize(cjson_compares);
    benchmarks = malloc(benchmark_num*sizeof(char*));
    params = malloc(param_num*sizeof(char*));
    compares = malloc(compare_num*sizeof(char*));
    space = malloc((param_num+1)*sizeof(void*));
    ret = malloc(compare_num*sizeof(void*));
    for (int i = 0; i < benchmark_num; ++i){
        cjson_benchmark_item = cJSON_GetArrayItem(cjson_benchmarks, i);
        benchmarks[i] = cjson_benchmark_item->valuestring;
    }
    for (int i = 0; i < param_num; ++i){
        cjson_param_item = cJSON_GetArrayItem(cjson_params, i);
        params[i] = cjson_param_item->valuestring;
    }
    for (int i = 0; i < compare_num; ++i){
        cjson_compare_item = cJSON_GetArrayItem(cjson_compares, i);
        compares[i] = cjson_compare_item->valuestring;
    }
    test_run(interface);
    for (int i = 0; i < benchmark_num; ++i){
        int compare_flag = 1;
        cjson_single_benchmark = cJSON_GetObjectItem(cjson_benchmark, benchmarks[i]);
        load_params(cjson_single_benchmark, params, space, param_num);
        cjson_benchmark_value = cJSON_GetObjectItem(cjson_single_benchmark, "benchmark");
        if (0 == strcmp(type, "ops")){
#ifdef GPU
            call_cu_ops(interface, space, ret);
#else
            call_ops(interface, space, ret);
#endif
        } else if (0 == strcmp(type, "graph")){
#ifdef GPU
            call_cu_graph(interface, space, ret);
#else
            call_graph(interface, space, ret);
#endif
        }
        compare_flag = compare_test(cjson_benchmark_value, ret, compares, compare_num);
        if (compare_flag == 0){
            test_msg_pass(benchmarks[i]);
        }
        else{
            all_pass = 1;
            test_msg_error(benchmarks[i]);
        }
    }
    test_res(all_pass, " ");
}

void run_all_benchmarks(char *benchmarks)
{
    FILE *fp = fopen(benchmarks, "r");
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
            run_benchmarks(tmp+head);
            head = i+1;
        }
    }
}

void run_unit_test(char *interface)
{
    if (0 == strcmp(interface, "add_bias")){
        run_benchmarks("./lumos_t/benchmark/core/ops/bias/add_bias.json");
    } else if (0 == strcmp(interface, "add_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/add_cpu.json");
    } else if (0 == strcmp(interface, "fill_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/fill_cpu.json");
    } else if (0 == strcmp(interface, "matrix_add_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/matrix_add_cpu.json");
    } else if (0 == strcmp(interface, "matrix_divide_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/matrix_divide_cpu.json");
    } else if (0 == strcmp(interface, "matrix_multiply_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/matrix_multiply_cpu.json");
    } else if (0 == strcmp(interface, "matrix_subtract_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/matrix_subtract_cpu.json");
    } else if (0 == strcmp(interface, "max_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/max_cpu.json");
    } else if (0 == strcmp(interface, "mean_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/mean_cpu.json");
    } else if (0 == strcmp(interface, "min_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/min_cpu.json");
    } else if (0 == strcmp(interface, "multy_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/multy_cpu.json");
    } else if (0 == strcmp(interface, "one_hot_encoding")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/one_hot_encoding.json");
    } else if (0 == strcmp(interface, "saxpy_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/saxpy_cpu.json");
    } else if (0 == strcmp(interface, "sum_channel_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/sum_channel_cpu.json");
    } else if (0 == strcmp(interface, "sum_cpu")){
        run_benchmarks("./lumos_t/benchmark/core/ops/cpu/sum_cpu.json");
    } else if (0 == strcmp(interface, "gemm")){
        run_benchmarks("./lumos_t/benchmark/core/ops/gemm/gemm.json");
    } else if (0 == strcmp(interface, "gemm_nn")){
        run_benchmarks("./lumos_t/benchmark/core/ops/gemm/gemm_nn.json");
    } else if (0 == strcmp(interface, "gemm_nt")){
        run_benchmarks("./lumos_t/benchmark/core/ops/gemm/gemm_nt.json");
    } else if (0 == strcmp(interface, "gemm_tn")){
        run_benchmarks("./lumos_t/benchmark/core/ops/gemm/gemm_tn.json");
    } else if (0 == strcmp(interface, "gemm_tt")){
        run_benchmarks("./lumos_t/benchmark/core/ops/gemm/gemm_tt.json");
    } else if (0 == strcmp(interface, "col2im")){
        run_benchmarks("./lumos_t/benchmark/core/ops/im2col/col2im.json");
    } else if (0 == strcmp(interface, "im2col")){
        run_benchmarks("./lumos_t/benchmark/core/ops/im2col/im2col.json");
    } else if (0 == strcmp(interface, "avgpool_gradient")){
        run_benchmarks("./lumos_t/benchmark/core/ops/pooling/avgpool_gradient.json");
    } else if (0 == strcmp(interface, "avgpool")){
        run_benchmarks("./lumos_t/benchmark/core/ops/pooling/avgpool.json");
    } else if (0 == strcmp(interface, "maxpool_gradient")){
        run_benchmarks("./lumos_t/benchmark/core/ops/pooling/maxpool_gradient.json");
    } else if (0 == strcmp(interface, "maxpool")){
        run_benchmarks("./lumos_t/benchmark/core/ops/pooling/maxpool.json");
    }
}

void release_params_space(void **space, int num)
{
    for (int i = 0; i < num; ++i){
        free(space[i]);
    }
}
