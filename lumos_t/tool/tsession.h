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

void run_benchmarks(char *benchmark);

#ifdef __cplusplus
}
#endif

#endif
