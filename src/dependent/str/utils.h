#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *fgetl(FILE *fp);
void strip(char *line);
char **split(char *line, char c, int *num);

char *inten2str(int x);
char *int2str(int x);
char *float2str(float x);

#endif