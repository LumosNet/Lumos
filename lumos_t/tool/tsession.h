#ifndef TSESSION_H
#define TSESSION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"
#include "utest.h"
#include "call.h"
#include "benchmark_json.h"
#include "compare.h"

#ifdef  __cplusplus
extern "C" {
#endif

// 返回值只有一个，默认拥有返回值，ret最后一个恒定为返回值
// 当release内存时，params全部release，ret最后一个release
void run_benchmarks(char *benchmark);
void run_all_benchmarks(char *benchmarks);

void release_params_space(void **space, int num);

#ifdef __cplusplus
}
#endif

#endif
