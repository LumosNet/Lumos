#ifndef XOR_H
#define XOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void xor(char *type, char *path);
void xor_detect(char *type, char *path);

#ifdef __cplusplus
}
#endif
#endif
