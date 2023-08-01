#ifndef RUN_TEST_H
#define RUN_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"

#include "analysis_benchmark_file.h"
#include "call.h"
#include "utest.h"
#include "compare.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define CPU 0
#define GPU 1
#define ALL 2

int run_by_benchmark_file(char *path, int coretype);

void run_all(char *listpath, int coretype);
void run_all_cases(char *listpath, int flag);
void run_by_interface(char *interface, int coretype);

#ifdef __cplusplus
}
#endif

#endif
