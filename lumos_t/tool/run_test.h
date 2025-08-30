#ifndef RUN_TEST_H
#define RUN_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"

#include "analysis_benchmark_file.h"
#include "utest.h"
#include "compare.h"
#include "logging.h"

#include "text_f.h"

#include "bias_call.h"
#include "cpu_call.h"
#include "gemm_call.h"
#include "im2col_call.h"
#include "image_call.h"
#include "pooling_call.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define CPU 0
#define GPU 1
#define ALL 2

int run_by_benchmark_file(char *path, TestInterface FUNC, int coretype, FILE *logfp);
int run_all_benchmark(int coretype, FILE *logfp);
TestInterface get_interface(char *name);

#ifdef __cplusplus
}
#endif

#endif
