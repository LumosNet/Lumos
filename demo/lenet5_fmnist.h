#ifndef LENET5_FMNIST_H
#define LENET5_FMNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void lenet5_fmnist(char *type, char *path);
void lenet5_fmnist_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
