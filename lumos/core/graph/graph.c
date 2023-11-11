#include "graph.h"

Graph *create_graph(int width, int height, int channel)
{
    Graph *graph = malloc(sizeof(Graph));
    graph->width = width;
    graph->height = height;
    graph->channel = channel;
    graph->layers = calloc(MAXLAYERS, sizeof(struct Layer*));
    return graph;
}

void append_layer2grpah(Graph *graph, Layer *l)
{
    for (int i = 0; i < MAXLAYERS; ++i){
        if (graph->layers[i] == NULL){
            graph->layers[i] = l;
            break;
        }
    }
}

void init_graph(Graph *g)
{
    Layer *l;
    int w = g->width, h = g->height, c = g->channel;
    for (int i = 0; i < MAXLAYERS; ++i){
        if (g->layers[i] == NULL) break;
        l = g->layers[i];
        l->subdivision = g->subdivision;
        l->init(l, w, h, c);
        w = l->output_w;
        h = l->output_h;
        c = l->output_c;
    }
}
