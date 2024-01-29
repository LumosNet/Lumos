#ifndef TEXT_F_H
#define TEXT_F_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "str_ops.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct Lines{
    char *line;
    struct Lines *next;
} Lines;

char *fgetl(FILE *fp);
char **fgetls(FILE *fp);

void fputl(FILE *fp, char *line);
void fputls(FILE *fp, char **lines, int n);

void **load_label_txt(char *path);

char *fget(char *file);

#ifdef __cplusplus
}
#endif

#endif