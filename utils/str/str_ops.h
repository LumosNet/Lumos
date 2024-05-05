#ifndef STR_OPS_H
#define STR_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef  __cplusplus
extern "C" {
#endif

void strip(char *line, char c);
int *split(char *line, char c);
void padding_string(char *space, char *str, int index);

char *inten2str(int x);
char *int2str(int x);

#ifdef __cplusplus
}
#endif

#endif