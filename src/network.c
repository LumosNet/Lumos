#include "network.h"

Network *load_network(char *cfg)
{
    NetParams *p = load_data_cfg("./cfg/lumos.cfg");
    Network *net = create_network(p);
    Node *n = p->head;
    int index = 0;
    while (n){
        LayerParams *l = n->val;
        Layer *layer = create_layer(l);
        net->layers[index] = layer;
        index += 1;
        n = n->next;
    }
    return net;
}

Network *create_network(NetParams *p)
{
    Network *net = malloc(sizeof(Network));
    Node *n = p->head;
    LayerParams *lp = n->val;
    n = lp->head;
    while (n){
        Params *pa = n->val;
        if (0 == strcmp(pa->key, "batch")){
            net->batch = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "width")){
            net->width = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "height")){
            net->height = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "learning_rate")){
            net->learning_rate = atoi(pa->val);
        }
        n = n->next;
    }
    net->n = -1;
    n = p->head;
    while (n){
        net->n += 1;
        n = n->next;
    }
    net->layers = malloc(sizeof(struct Layer*));
    return net;
}

Layer *create_layer(LayerParams *p)
{
    Layer *layer = malloc(sizeof(Layer));
    if (0 == strcmp(p->type, "convolutional")){
        layer->type = CONVOLUTIONAL;
    } else if (0 == strcmp(p->type, "pool")){
        layer->type = POOLING;
    }  else if (0 == strcmp(p->type, "active")){
        layer->type = ACTIVE;
    } else if (0 == strcmp(p->type, "fullconnect")){
        layer->type = FULLCONNECT;
    }
    Node *n = p->head;
    while (n){
        Params *pa = n->val;
        if (0 == strcmp(pa->key, "filters")){
            layer->filters = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "ksize")){
            layer->ksize = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "stride")){
            layer->stride = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "pad")){
            layer->pad = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "activation")){
            Activation active = load_activate_type(pa->val);
            layer->active = load_activate(active);
            layer->gradient = load_gradient(active);
        } else if (0 == strcmp(pa->key, "type")){
            layer->pool = load_pooling_type(pa->key);
        }
        n = n->next;
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