#ifndef NETWORK_H
#define NETWORK_H

#include "lumos.h"
#include "parser.h"

#ifdef __cplusplus
extern "C" {
#endif

Network *load_network(char **cfg);
void train(Network *net);
void forward_network(Network *net);
void backward_network(Network *net);

#ifdef __cplusplus
}
#endif

#endif