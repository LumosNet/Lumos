#include "graph.h"

Graph *create_graph(char *name, int layer_n)
{
    Graph *graph = malloc(sizeof(Graph));
    graph->graph_name = name;
    graph->layer_list_num = layer_n;
    graph->layer_num = 0;
    graph->layers = calloc(layer_n, sizeof(Layer));
    fprintf(stderr, "[%s]         max   %d  Layers\n", graph->graph_name, graph->layer_list_num);
    return graph;
}

void append_layer2grpah(Graph *graph, Layer *l)
{
    graph->layers[graph->layer_num] = l;
    graph->layer_num += 1;
    l->index = graph->layer_num;
}

void init_graph(Graph *g, int w, int h, int c)
{
    fprintf(stderr, "\nStart To Init Graph\n");
    fprintf(stderr, "[%s]                     Inputs         Outputs\n", g->graph_name);
    Layer *l;
    for (int i = 0; i < g->layer_num; ++i)
    {
        l = g->layers[i];
        switch (l->type)
        {
        case AVGPOOL:
            init_avgpool_layer(l, w, h, c);
            break;
        case CONNECT:
            init_connect_layer(l, w, h, c);
            break;
        case CONVOLUTIONAL:
            init_convolutional_layer(l, w, h, c);
            break;
        case IM2COL:
            init_im2col_layer(l, w, h, c);
            break;
        case MAXPOOL:
            init_maxpool_layer(l, w, h, c);
            break;
        case MSE:
            init_mse_layer(l, w, h, c);
            break;
        case SOFTMAX:
            init_softmax_layer(l, w, h, c);
            break;
        case DROPOUT:
            init_dropout_layer(l, w, h, c);
            break;
        default:
            break;
        }
        w = l->output_w;
        h = l->output_h;
        c = l->output_c;
    }
}
