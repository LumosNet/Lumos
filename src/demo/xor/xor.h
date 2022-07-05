#ifndef XOR_H
#define XOR_H

#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "manager.h"
#include "dispatch.h"
#include "graph.h"
#include "layer.h"
#include "connect_layer.h"
#include "im2col_layer.h"
#include "mse_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void xor();

#ifdef __cplusplus
}
#endif

#endif