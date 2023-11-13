#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"

#include "avgpool_layer.h"
#include "maxpool_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "im2col_layer.h"
#include "mse_layer.h"

#include "avgpool_layer_gpu.h"
#include "maxpool_layer_gpu.h"
#include "connect_layer_gpu.h"
#include "convolutional_layer_gpu.h"
#include "im2col_layer_gpu.h"
#include "mse_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAXLAYERS 1024

typedef struct graph{
    int *command;
    int num;
    int subdivision;
    float learning_rate;
    int width;
    int height;
    int channel;
    float *input;
    float *output;
    float *delta;
    float *workspace;
    float *truth;
    Layer **layers;

    char **data_path;
    char **truth_path;
    int data_num;
} graph, Graph;

Graph *create_graph(int width, int height, int channel);
void append_layer2grpah(Graph *graph, Layer *l);
void init_graph(Graph *g, int num, char *data_paths, char *truth_paths);

void create_input(Graph *g);
void create_workspace(Graph *g);
void create_truthspace(Graph *g, int num);
void bind_workspace(Graph *g);
void bind_truthspace(Graph *g);
void bind_data_paths(Graph *g, char *data_paths);
void bind_label_paths(Graph *g, char *truth_paths);

void forward_graph(Graph *g);
void backward_graph(Graph *g);
void update_graph(Graph *g);

void train(Graph *g);
void detect(Graph *g);

#ifdef __cplusplus
}
#endif

#endif