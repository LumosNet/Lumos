#ifndef GRAPH_H
#define GRAPH_H

#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct graph{
    int layer_num;
    int height;
    int channel;
    float *input;
    float *output;
    float *delta;
    Layer *layers;

    int kinds;
    char **label;
    // Label *labels;

    int num;
    char **data;

    CFG *cfg;
} graph, Graph;

Graph load_graph(char *cfg_path);
Graph create_graph(CFGPiece *p, int layer_n);

void init_graph(Graph g, int w, int h, int c);
void restore_graph(Graph g);

#ifdef __cplusplus
}
#endif

#endif