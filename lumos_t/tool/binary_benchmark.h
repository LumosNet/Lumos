#ifndef BINARY_BENCHMARK_H
#define BINARY_BENCHMARK_H

#include <stdio.h>
#include <stdlib.h>

#include "bias.h"

#ifdef  __cplusplus
extern "C" {
#endif

void *get_binary_benchmark(FILE *fp);

int analysis_interface(void *buffer, int offset); //返回接口标识
int analysis_usecase(void *buffer, int offset); //返回用例参数个数
void analysis_parameters(void *buffer, int offset, void **parameters);

#ifdef __cplusplus
}
#endif

#endif
