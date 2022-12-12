#ifndef TSESSION_H
#define TSESSION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "cJSON_Utils.h"
#include "benchmark_json.h"

#ifdef  __cplusplus
extern "C" {
#endif

void do_single_interface_test(char **benchmark, char **bnames);

#ifdef __cplusplus
}
#endif

#endif
