#ifndef LENET5_CIFAR_H
#define LENET5_CIFAR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void lenet5_cifar(char *type, char *path);
void lenet5_cifar_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
