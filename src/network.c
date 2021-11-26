#include "network.h"

Network *load_network(char *cfg)
{
    NetParams *p = load_data_cfg(cfg);
    Node *n = p->head;
    Network *net = create_network(n->val, p->size);
    n = n->next;
    int h = net->height;
    int w = net->width;
    int c = net->channel;
    int index = 0;
    while (n){
        LayerParams *l = n->val;
        fprintf(stderr, "  %d  ", index+1);
        Layer *layer = create_layer(l, h, w, c);
        h = layer->output_h;
        w = layer->output_w;
        c = layer->output_c;
        net->layers[index] = layer;
        index += 1;
        n = n->next;
    }
    return net;
}

Network *create_network(LayerParams *p, int size)
{
    Network *net = malloc(sizeof(Network));
    fprintf(stderr, "%s\n", p->type);
    Node *n = p->head;
    while (n){
        Params *pa = n->val;
        if (0 == strcmp(pa->key, "batch")){
            net->batch = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "width")){
            net->width = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "height")){
            net->height = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "channel")){
            net->channel = atoi(pa->key);
        } else if (0 == strcmp(pa->key, "learning_rate")){
            net->learning_rate = atof(pa->val);
        }
        n = n->next;
    }
    net->n = size;
    net->layers = malloc(size*sizeof(struct Layer*));
    fprintf(stderr, "index  type   filters   ksize        input                  output\n");
    return net;
}

Layer *create_layer(LayerParams *p, int h, int w, int c)
{
    Layer *layer;
    if (0 == strcmp(p->type, "convolutional")){
        layer = make_convolutional_layer(p, h, w, c);
    } else if (0 == strcmp(p->type, "pooling")){
        layer = make_pooling_layer(p, h, w, c);
    } else if (0 == strcmp(p->type, "softmax")){
        layer = make_softmax_layer(p, h, w, c);
    } else if (0 == strcmp(p->type, "connect")){
        layer = make_connect_layer(p, h, w, c);
    } else if (0 == strcmp(p->type, "activation")){
        layer = make_activation_layer(p, h, w, c);
    }
    return layer;
}

void add_layer2net(Network *net, Layer *l);

void forward_network(Network *net)
{

}

void backward_network(Network *net)
{

}