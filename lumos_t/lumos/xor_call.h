#ifndef XOR_CALL_H
#define XOR_CALL_H

#include <stdlib.h>
#include <stdio.h>

#include "cpu.h"
#include "graph.h"
#include "layer.h"
#include "im2col_layer.h"
#include "connect_layer.h"
#include "mse_layer.h"
#include "session.h"
#include "manager.h"
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void _xor_label2truth(char **label, float *truth);

void call_xor(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
