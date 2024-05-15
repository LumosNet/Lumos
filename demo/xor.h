#ifndef XOR_H
#define XOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "connect_layer.h"
#include "im2col_layer.h"
#include "mse_layer.h"
#include "graph.h"
#include "layer.h"
#include "session.h"

#ifdef __cplusplus
extern "C" {
#endif

void xor(char *type, char *path);
void xor_detect(char *type, char *path);

#ifdef __cplusplus
}
#endif
#endif
