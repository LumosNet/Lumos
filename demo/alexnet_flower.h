#ifndef ALEXNET_FLOWER_H
#define ALEXNET_FLOWER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void alexnet_flower(char *type, char *path);
void alexnet_flower_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
