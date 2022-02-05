#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fill_cpu_float(float *data, float x, int num);
float *flid_float_cpu(float *data, int h, int w, int c);
char *fgetl(FILE *fp);
void strip(char *line);
char **split(char *line, char c, int *num);

char *int2str(int x);
char *float2str(float x);
char *link_str(char *a, char *b);

char *inten2str(int x);

void debug_data(FILE *fp, int height, int width, float *data, char *str);
void debug_str(FILE *fp, char *str);
#endif