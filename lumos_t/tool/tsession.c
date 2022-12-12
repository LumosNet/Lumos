#include "tsession.h"

void run_benchmarks(char **benchmark, char **bnames)
{
    cJSON *cjson_benchmark = NULL;
    cJSON *cjson_public = NULL;
    cJSON *cjson_interface = NULL;
    cJSON *cjson_type = NULL;
    cJSON *cjson_benchmarks = NULL;
    cJSON *cjson_params = NULL;
    cJSON *cjson_compare = NULL;
    char *interface = NULL;
    char *type = NULL;
    char **benchmarks = NULL;
    char **params = NULL;
    char *compare = NULL;
    int benchmark_num = 0;
    int param_num = 0;
    char *json_benchmark = load_from_json_file(benchmark);
    cjson_benchmark = cJSON_Parse(json_benchmark);
    cjson_public = cJSON_GetObjectItemA(cjson_benchmark, "Public");
    cjson_interface = cJSON_GetObjectItemA(cjson_public, "interface");
    cjson_type = cJSON_GetObjectItemA(cjson_public, "type");
    cjson_benchmarks = cJSON_GetObjectItemA(cjson_public, "benchmarks");
    cjson_params = cJSON_GetObjectItemA(cjson_public, "params");
    cjson_compare = cJSON_GetObjectItemA(cjson_public, "compare");
}
