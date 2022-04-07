#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include <string.h>

#include "lumos.h"
#include "parser.h"
#include "im2col.h"

#include "debug.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_maxpool_layer(Layer l, Network net);
void backward_maxpool_layer(Layer l, Network net);


#ifdef __cplusplus
}
#endif

#endif