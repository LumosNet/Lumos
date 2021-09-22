#ifndef NETWORK_H
#define NETWORK_H

#include <string.h>

#include "tensor.h"
#include "parser.h"
#include "utils.h"
#include "active.h"
#include "pooling_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

Network *load_network(char *cfg);
void train(Network *net);
void forward_network(Network *net);
void backward_network(Network *net);

Network *create_network(NetParams *p);
Layer *create_layer(LayerParams *p);
void add_layer2net(Network *net, Layer *l);

#ifdef __cplusplus
}
#endif

#endif