#ifndef GRAPH_H
#define GRAPH_H

#include "layer.h"
#include "cfg_f.h"

#include "avgpool_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "im2col_layer.h"
#include "maxpool_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct graph{
    char *graph_name;
    int layer_list_num;
    int layer_num;
    int height;
    int channel;
    float *input;
    float *output;
    float *delta;
    Layer *layers;

    int kinds;
    char **label;

    int num;
    char **data;

    CFG *cfg;
} graph, Graph;

Graph create_graph(char *name, int layer_n);
Graph create_graph_by_cfg(CFGPiece *p, int layer_n);
Graph load_graph_from_cfg(char *cfg_path);
Layer create_layer_by_cfg(CFGPiece *p);

void append_layer2grpah(Graph graph, Layer l);

void init_graph(Graph g, int w, int h, int c);
void restore_graph(Graph g);

#ifdef __cplusplus
}
#endif

#endif