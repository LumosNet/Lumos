#include "network.h"

Graph *load_graph(char *cfg)
{
    CFG *cfg = load_conf_cfg(cfg);
    CFGPieces *pieces = cfg->pieces;
    CFGPiece *net_params = pieces->head;
    Graph *graph = create_graph(net_params, cfg->piece_num-1);
    CFGPiece *piece = net_params->next;
    int index = 0;
    while (piece){
        Layer layer = create_layer(piece);
        graph->layers[index] = layer;
        index += 1;
        piece = piece->next;
    }
    return graph;
}

Graph *create_graph(CFGPiece *p, int layer_n)
{
    Graph *graph = malloc(sizeof(struct Graph));
    CFGParams *params = p->params;
    CFGParam *param = params->head;
    while (param){
        param = param->next;
    }
    graph->layer_num = layer_n;
    graph->layers = calloc(layer_n, sizeof(struct Layer));
    return graph;
}

void forward_network(Network *net)
{
    for (int i = 0; i < net->n; ++i){
        Layer *l = &net->layers[i];
        l->input = net->output;
        l->forward(l[0], net[0]);
        net->output = l->output;
        fill_cpu(net->workspace, net->workspace_size, 0, 1);
    }
}

void backward_network(Network *net)
{
    net->delta = NULL;
    for (int i = net->n-1; i >= 0; --i){
        Layer *l = &net->layers[i];
        l->backward(l[0], net[0]);
        net->delta = l->delta;
        fill_cpu(net->workspace, net->workspace_size, 0, 1);
        fill_cpu(l->output, l->outputs, 0, 1);
    }
}
