#ifndef TEXT_F_H
#define TEXT_F_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "str_ops.h"

#ifdef  __cplusplus
extern "C" {
#endif

void fputl(FILE *fp, char *line);
void fputls(FILE *fp, char **lines, int n);
char *fget(char *file);
int lines_num(char *tmp);
int format_str(char *tmp);
int format_str_float(char *tmp);

#ifdef __cplusplus
}
#endif

#endif