#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct node Node;

typedef struct graph{
    float *input;
    float *output;
    float *delta;
    Node *head;
    Node *tail;
} graph, Graph;

struct node{
    Layer *l;
    Node *head;
    Node *next;
};

Graph *create_graph();

void append_layer2grpah(Graph *graph, Layer *l);
void init_graph(Graph *g, int w, int h, int c, int coretype);
void set_graph(Graph *g, float *space, float *truth, float *loss);
void forward_graph(Graph *g, float *input, int coretype, int subdivision);
void backward_graph(Graph *g, float rate, int coretype, int subdivision);
void update_graph(Graph *g, int coretype);

#ifdef __cplusplus
}
#endif

#endif