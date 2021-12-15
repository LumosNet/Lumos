#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include <string.h>

#include "lumos.h"
#include "parser.h"
#include "im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_avgpool_layer(Layer l, Network net);
void backward_avgpool_layer(Layer l, Network net);


#ifdef __cplusplus
}
#endif

#endif