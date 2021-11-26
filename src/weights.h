#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void load_weights(Network *net, char *path);
void save_weights(Network *net, char *path);

#ifdef __cplusplus
}
#endif

#endif