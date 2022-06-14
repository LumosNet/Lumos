#include "graph.h"

Graph load_graph(char *cfg_path)
{
    CFG *cfg = load_conf_cfg(cfg_path);
    CFGPieces *pieces = cfg->pieces;
    CFGPiece *net_params = pieces->head;
    Graph graph = create_graph(net_params, cfg->piece_num-1);
    CFGPiece *piece = net_params->next;
    int index = 0;
    while (piece){
        fprintf(stderr, "%3d ", index+1);
        Layer layer = create_layer(piece);
        graph.layers[index] = layer;
        index += 1;
        piece = piece->next;
    }
    return graph;
}

Graph create_graph(CFGPiece *p, int layer_n)
{
    Graph graph = {0};
    CFGParams *params = p->params;
    CFGParam *param = params->head;
    graph.graph_name = p->name;
    while (param){
        param = param->next;
    }
    graph.layer_num = layer_n;
    graph.layers = calloc(layer_n, sizeof(Layer));
    fprintf(stderr, "[%s]       %d  layers\n", graph.graph_name, graph.layer_num);
    return graph;
}

Layer create_layer(CFGPiece *p)
{
    Layer layer;
    if (0 == strcmp(p->name, "convolutional")){
        layer = make_convolutional_layer(p->params);
    } else if (0 == strcmp(p->name, "avgpool")){
        layer = make_avgpool_layer_by_cfg(p->params);
    } else if (0 == strcmp(p->name, "maxpool")){
        layer = make_maxpool_layer(p->params);
    } else if (0 == strcmp(p->name, "connect")){
        layer = make_connect_layer(p->params);
    } else if (0 == strcmp(p->name, "im2col")){
        layer = make_im2col_layer(p->params);
    }
    return layer;
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