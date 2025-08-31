#ifndef ALEXNET_XRAY_H
#define ALEXNET_XRAY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void alexnet_xray(char *type, char *path);
void alexnet_xray_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif