#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"

#include "avgpool_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "im2col_layer.h"
#include "maxpool_layer.h"

#include "mse_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAXLAYERS 1024

typedef struct graph{
    int subdivision;
    int width;
    int height;
    int channel;
    float *input;
    float *output;
    float *delta;
    float *truth;
    Layer **layers;
} graph, Graph;

Graph *create_graph(int width, int height, int channel);

void append_layer2grpah(Graph *graph, Layer *l);
void init_graph(Graph *g);

#ifdef __cplusplus
}
#endif

#endif