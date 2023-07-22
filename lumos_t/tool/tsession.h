#ifndef TSESSION_H
#define TSESSION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"
#include "str_ops.h"
#include "analysis_benchmark_file.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct session_t SessionT;
typedef struct benchmark Benchmark;
typedef struct compare Compare;

struct session_t{
    char *interface;
    char **cases;
    char **params;
    char **compares;
    int case_num;
    int param_num;
    int compare_num;
    Benchmark *benchs;
    Compare *comps;
};

struct benchmark{
    char **type;
    void **value;
};

struct compare{
    char **type;
    void **value;
};

SessionT *make_t_session(char *path);

#ifdef __cplusplus
}
#endif

#endif
