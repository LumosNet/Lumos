#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void save_data(float *data, int c, int h, int w, int batch, char *file);

#ifdef __cplusplus
}
#endif

#endif