#ifndef LENET5_CIFAR10_H
#define LENET5_CIFAR10_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void lenet5_cifar10(char *type, char *path);
void lenet5_cifar10_detect(char *type, char *path);

#ifdef __cplusplus
}
#endif
#endif
