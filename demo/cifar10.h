#ifndef CIFAR10_H
#define CIFAR10_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void cifar10(char *type, char *path);
void cifar10_detect(char *type, char *path);

#ifdef __cplusplus
}
#endif
#endif
