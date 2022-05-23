#ifndef NETWORK_H
#define NETWORK_H

#include <string.h>

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

struct network;
typedef struct network network;

struct label;
typedef struct label label;

typedef struct label{
    float *data;
    int num;
    struct label *next;
} Label;

typedef struct network{
    int n;
    int batch;
    int width;
    int height;
    int channel;
    float learning_rate;
    size_t workspace_size;
    float *workspace;
    float *input;
    float *output;
    float *delta;
    Layer *layers;

    int kinds;
    char **label;
    Label *labels;

    int num;
    char **data;
} network, Network, NetWork;

Network *load_network(char *cfg);
void train(Network *net, int x);
void test(Network *net, char *test_png, char *test_label);
void init_network(Network *net, char *data_file, char *weight_file);
void forward_network(Network *net);
void backward_network(Network *net);

Network *create_network(LayerParams *p, int size);

#ifdef __cplusplus
}
#endif

#endif