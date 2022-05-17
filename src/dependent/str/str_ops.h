#ifndef STR_OPS_H
#define STR_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef  __cplusplus
extern "C" {
#endif

void strip(char *line);
char **split(char *line, char c, int *num);

char *inten2str(int x);
char *int2str(int x);
char *float2str(float x);

#ifdef __cplusplus
}
#endif

#endif