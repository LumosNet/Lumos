#ifndef PROGRESS_BAR_API
#define PROGRESS_BAR_API

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void progress_bar(int n, int m, double time, float loss);

#ifdef __cplusplus
}
#endif

#endif