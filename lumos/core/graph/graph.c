#include "graph.h"

Graph *create_graph()
{
    Graph *graph = malloc(sizeof(Graph));
    graph->head = NULL;
    graph->tail = NULL;
    fprintf(stderr, "[Lumos]         Module Structure\n");
    return graph;
}

void append_layer2grpah(Graph *graph, Layer *l)
{
    Node *layer = malloc(sizeof(Node));
    if (graph->tail){
        Node *tail = graph->tail;
        tail->next = layer;
    }
    layer->l = l;
    layer->next = NULL;
    layer->head = graph->tail;
    graph->tail = layer;
    if (graph->head == NULL) graph->head = layer;
}

void init_graph(Graph *g, int w, int h, int c, int coretype, int subdivision, char *weights_path)
{
    fprintf(stderr, "\nStart To Init Graph\n");
    fprintf(stderr, "[Lumos]                     Inputs         Outputs\n");
    Node *layer = g->head;
    Layer *l;
    FILE *fp = NULL;
    if (weights_path){
        fp = fopen(weights_path, "rb");
    }
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU){
                l->initializegpu(l, w, h, c, subdivision);
                if (l->weightinitgpu) l->weightinitgpu(*l, fp);
            } else {
                l->initialize(l, w, h, c, subdivision);
                if (l->weightinit) l->weightinit(*l, fp);
            }
        } else {
            break;
        }
        layer = layer->next;
        w = l->output_w;
        h = l->output_h;
        c = l->output_c;
    }
    if (fp){
        fclose(fp);
    }
}

void set_graph(Graph *g, float *space, float *truth, float *loss)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            l->truth = truth;
            l->loss = loss;
            l->workspace = space;
        } else {
            break;
        }
        layer = layer->next;
    }
}

void forward_graph(Graph *g, float *input, int coretype, int subdivision)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            l->status = g->status;
            l->input = input;
            if (coretype == GPU){
                l->forwardgpu(*l, subdivision);
            } else {
                l->forward(*l, subdivision);
            }
        } else {
            break;
        }
        layer = layer->next;
        input = l->output;
    }
}

void backward_graph(Graph *g, float rate, int coretype, int subdivision)
{
    Node *layer = g->tail;
    Layer *l;
    float *n_delta;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU){
                l->backwardgpu(*l, rate, subdivision, n_delta);
            } else {
                l->backward(*l, rate, subdivision, n_delta);
            }
        } else {
            break;
        }
        layer = layer->head;
        n_delta = l->delta;
    }
}

void update_graph(Graph *g, int coretype)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU && l->updategpu) l->updategpu(*l);
            if (coretype == CPU && l->update) l->update(*l);
        } else {
            break;
        }
        layer = layer->next;
    }
}

void save_weights(Graph *g, int coretype, FILE *fp)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU && l->saveweightsgpu) l->saveweightsgpu(*l, fp);
            if (coretype == CPU && l->saveweights) l->saveweights(*l, fp);
        } else {
            break;
        }
        layer = layer->next;
    }
}

void free_graph(Graph *g, int coretype)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU && l->freelayergpu) l->freelayergpu(*l);
            if (coretype == CPU && l->freelayer) l->freelayer(*l);
        } else {
            break;
        }
        layer = layer->next;
    }
}
