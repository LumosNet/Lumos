#include "graph.h"

Graph create_graph(char *name, int layer_n)
{
    Graph graph = {0};
    graph.graph_name = name;
    graph.layer_list_num = layer_n;
    graph.layer_num = 0;
    graph.layers = calloc(layer_n, sizeof(Layer));
    fprintf(stderr, "[%s]         max   %d  Layers\n", graph.graph_name, graph.layer_list_num);
    return graph;
}

Graph create_graph_by_cfg(CFGPiece *p, int layer_n)
{
    Graph graph = create_graph(p->name, layer_n);
    return graph;
}

Graph load_graph_from_cfg(char *cfg_path)
{
    fprintf(stderr, "Load Graph From CFG File '%s'\n", cfg_path);
    CFG *cfg = load_conf_cfg(cfg_path);
    CFGPieces *pieces = cfg->pieces;
    CFGPiece *net_params = pieces->head;
    Graph graph = create_graph_by_cfg(net_params, cfg->piece_num-1);
    CFGPiece *piece = net_params->next;
    int index = 0;
    while (piece){
        fprintf(stderr, "%3d ", index+1);
        Layer *layer = create_layer_by_cfg(piece);
        graph.layers[index] = layer;
        graph.layer_num += 1;
        index += 1;
        piece = piece->next;
    }
    fprintf(stderr, "Load Graph From CFG File '%s' Succeed\n", cfg_path);
    return graph;
}

Layer create_layer_by_cfg(CFGPiece *p)
{
    Layer *layer;
    if (0 == strcmp(p->name, "convolutional")){
        layer = make_convolutional_layer_by_cfg(p->params);
    } else if (0 == strcmp(p->name, "avgpool")){
        layer = make_avgpool_layer_by_cfg(p->params);
    } else if (0 == strcmp(p->name, "maxpool")){
        layer = make_maxpool_layer_by_cfg(p->params);
    } else if (0 == strcmp(p->name, "connect")){
        layer = make_connect_layer_by_cfg(p->params);
    } else if (0 == strcmp(p->name, "im2col")){
        layer = make_im2col_layer_by_cfg(p->params);
    }
    return layer;
}

void append_layer2grpah(Graph graph, Layer l)
{
    graph.layers[graph.layer_num] = l;
    graph.layer_num += 1;
}

void init_graph(Graph g, int w, int h, int c)
{
    Layer l;
    for (int i = 0; i < g.layer_num; ++i){
        l = g.layers[i];
        switch (l.type){
            case AVGPOOL:
                init_avgpool_layer(l, w, h, c); break;
            case CONNECT:
                init_connect_layer(l, w, h, c); break;
            case CONVOLUTIONAL:
                init_convolutional_layer(l, w, h, c); break;
            case IM2COL:
                init_im2col_layer(l, w, h, c); break;
            case MAXPOOL:
                init_maxpool_layer(l, w, h, c); break;
            default:
                break;
        }
    }
}

void restore_graph(Graph g)
{
    Layer l;
    for (int i = 0; i < g.layer_num; ++i){
        l = g.layers[i];
        switch (l.type){
            case AVGPOOL:
                restore_avgpool_layer(l); break;
            case CONNECT:
                restore_connect_layer(l); break;
            case CONVOLUTIONAL:
                restore_convolutional_layer(l); break;
            case IM2COL:
                restore_im2col_layer(l); break;
            case MAXPOOL:
                restore_maxpool_layer(l); break;
            default:
                break;
        }
    }
}