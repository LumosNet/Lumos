#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG   0
#define INFO    1
#define WARNING 2 
#define ERROR   3
#define FATAL   4

char* getDateTime();
void logging_type(int type, FILE *buffer);
void logging_msg(int type, char *msg, FILE *buffer);
void logging_data(char *type, void *data, int h, int w, int c, FILE *buffer);

void logging_int_data(int h, int w, int c, int *data, FILE *buffer);
void logging_float_data(int h, int w, int c, float *data, FILE *buffer);

#ifdef __cplusplus
}
#endif
#endif
