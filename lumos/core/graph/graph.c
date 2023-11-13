#include "graph.h"

Graph *create_graph(int width, int height, int channel)
{
    Graph *graph = malloc(sizeof(Graph));
    graph->width = width;
    graph->height = height;
    graph->channel = channel;
    graph->num = 0;
    graph->layers = calloc(MAXLAYERS, sizeof(struct Layer*));
    return graph;
}

void append_layer2grpah(Graph *graph, Layer *l)
{
    for (int i = 0; i < MAXLAYERS; ++i){
        if (graph->layers[i] == NULL){
            graph->layers[i] = l;
            graph->num += 1;
            break;
        }
    }
}

void init_graph(Graph *g, int num, char *data_paths, char *truth_paths)
{
    Layer *l;
    int w = g->width, h = g->height, c = g->channel;
    for (int i = 0; i < g->num; ++i){
        l = g->layers[i];
        l->subdivision = g->subdivision;
        l->init(l, w, h, c);
        w = l->output_w;
        h = l->output_h;
        c = l->output_c;
    }
    create_workspace(g);
    bind_workspace(g);
    create_truthspace(g, num);
    bind_truthspace(g);
    create_input(g);
    bind_data_paths(g, data_paths);
    bind_label_paths(g, truth_paths);
}

void create_input(Graph *g)
{
    g->input = calloc(g->width*g->height*g->channel*g->subdivision, sizeof(float));
}

void create_workspace(Graph *g)
{
    int size = -1;
    Layer *l;
    for (int i = 0; i < g->num; ++i){
        if (g->layers[i] == NULL) break;
        l = g->layers[i];
        if (l->workspace_size > size){
            size = l->workspace_size;
        }
    }
    g->workspace = calloc(size, sizeof(float));
}

void create_truthspace(Graph *g, int num)
{
    g->truth = calloc(num, sizeof(float));
}

void bind_workspace(Graph *g)
{
    Layer *l;
    for (int i = 0; i < g->num; ++i){
        if (g->layers[i] == NULL) break;
        l = g->layers[i];
        l->workspace = g->workspace;
    }
}

void bind_truthspace(Graph *g)
{
    Layer *l = g->layers[g->num-1];
    l->truth = g->truth;
}

void bind_data_paths(Graph *g, char *data_paths)
{
    char *tmp = fget(data_paths);
    int *index = split(tmp, '\n');
    int lines = index[0];
    g->data_num = lines;
    g->data_path = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        g->data_path[i] = tmp+index[i+1];
    }
    free(index);
}

void bind_label_paths(Graph *g, char *truth_paths)
{
    char *tmp = fget(truth_paths);
    int *index = split(tmp, '\n');
    int lines = index[0];
    g->truth_path = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        g->truth_path[i] = tmp+index[i+1];
    }
    free(index);
}

void forward_graph(Graph *g)
{
    Layer *l;
    float *input = g->input;
    l = g->layers[0];
    for (int i = 0; i < g->num; ++i){
        l = g->layers[i];
        l->input = input;
        l->forward(*l, g->subdivision);
        input = l->output;
    }
}

void backward_graph(Graph *g)
{
    Layer *l;
    float *delta = NULL;
    for (int i = g->num-1; i >= 0; --i){
        l = g->layers[i];
        l->backward(*l, g->subdivision, delta);
        delta = l->delta;
    }
}

void update_graph(Graph *g)
{
    Layer *l;
    for (int i = 0; i < g->num; ++i){
        l = g->layers[i];
        if (l->updateweights){
            l->updateweights(*l);
        }
    }
}

void train(Graph *g)
{
    
}

void detect(Graph *g)
{

}
