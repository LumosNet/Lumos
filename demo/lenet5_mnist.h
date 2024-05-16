#ifndef LENET5_MNIST_H
#define LENET5_MNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void lenet5_mnist(char *type, char *path);
void lenet5_mnist_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
