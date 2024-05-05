#ifndef DATA_LOG_H
#define DATA_LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void logging_data(char *type, int h, int w, int c, void *data, FILE *buffer);

void logging_int_data(int h, int w, int c, int *data, FILE *buffer);
void logging_float_data(int h, int w, int c, float *data, FILE *buffer);

#ifdef __cplusplus
}
#endif
#endif
