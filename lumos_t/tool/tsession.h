#ifndef TSESSION_H
#define TSESSION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"
#include "str_ops.h"
#include "utest.h"
#include "call.h"
#include "benchmark_json.h"
#include "compare.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define CPU 0
#define GPU 1

// 返回值只有一个，默认拥有返回值，ret最后一个恒定为返回值
// 当release内存时，params全部release，ret最后一个release
void run_benchmarks(char *benchmark, int coretype);
void run_all_benchmarks(char *benchmarks, int coretype);

void run_unit_test(char *interface);

void release_params_space(void **space, int num);

#ifdef __cplusplus
}
#endif

#endif
