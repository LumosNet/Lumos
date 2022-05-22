#ifndef NETWORK_H
#define NETWORK_H

#include <string.h>

#include "parser.h"
#include "str_ops.h"
#include "cfg_f.h"
#include "active.h"
#include "data.h"
#include "pooling_layer.h"
#include "convolutional_layer.h"
#include "softmax_layer.h"
#include "connect_layer.h"
#include "im2col_layer.h"
#include "mse_layer.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Network *load_network(char *cfg);
void train(Network *net, int x);
void test(Network *net, char *test_png, char *test_label);
void init_network(Network *net, char *data_file, char *weight_file);
void forward_network(Network *net);
void backward_network(Network *net);

Network *create_network(LayerParams *p, int size);
Layer create_layer(Network *net, LayerParams *p, int h, int w, int c);

#ifdef __cplusplus
}
#endif

#endif