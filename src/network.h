#ifndef NETWORK_H
#define NETWORK_H

#include <string.h>

#include "tensor.h"
#include "parser.h"
#include "utils.h"
#include "active.h"
#include "pooling_layer.h"
#include "convolutional_layer.h"
#include "softmax_layer.h"
#include "connect_layer.h"
#include "activation_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

Network *load_network(char *cfg);
void train(Network *net);
void forward_network(Network *net);
void backward_network(Network *net);

Network *create_network(LayerParams *p, int size);
Layer create_layer(Network *net, LayerParams *p, int h, int w, int c);

#ifdef __cplusplus
}
#endif

#endif