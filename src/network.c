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
        Layer layer = create_layer(net, l, h, w, c);
        h = layer.output_h;
        w = layer.output_w;
        c = layer.output_c;
        layer.i = index;
        net->layers[index] = layer;
        index += 1;
        n = n->next;
    }
    net->workspace = calloc(net->workspace_size, sizeof(float));
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
            net->channel = atoi(pa->val);
        } else if (0 == strcmp(pa->key, "learning_rate")){
            net->learning_rate = atof(pa->val);
        }
        n = n->next;
    }
    net->n = size-1;
    net->workspace_size = 0;
    net->layers = calloc(size, sizeof(Layer));
    net->input = calloc(net->batch*net->width*net->height*net->channel, sizeof(float));
    net->labels = calloc(net->batch, sizeof(Label));
    fprintf(stderr, "index  type   filters   ksize        input                  output\n");
    return net;
}

Layer create_layer(Network *net, LayerParams *p, int h, int w, int c)
{
    Layer layer;
    if (0 == strcmp(p->type, "convolutional")){
        layer = make_convolutional_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "pooling")){
        layer = make_pooling_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "softmax")){
        layer = make_softmax_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "connect")){
        layer = make_connect_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "im2col")){
        layer = make_im2col_layer(p, net->batch, h, w, c);
    }
    if (layer.workspace_size > net->workspace_size) net->workspace_size = layer.workspace_size;
    return layer;
}

void init_network(Network *net, char *data_file, char *weight_file)
{
    int *ln = malloc(sizeof(int));
    int *pn = malloc(sizeof(int));
    char **datas = read_lines(data_file, ln);
    char *data_path;
    char *label_path;
    for (int i = 0; i < ln[0]; ++i){
        char *line = datas[i];
        char **params = split(line, '=', pn);
        if (0 == strcmp(params[0], "classes")){
            net->kinds = atoi(params[1]);
        } else if(0 == strcmp(params[0], "data")){
            data_path = params[1];
        } else if(0 == strcmp(params[0], "label")){
            label_path = params[1];
        }
    }
    load_train_path(net, data_path, label_path);
    load_weights(net, weight_file);
}

void train(Network *net, int x)
{
    int offset = 0;
    int n = 0;
    while (1){
        load_train_data(net, offset);
        debug_str(net->fdebug, "\npush forward\n");
        forward_network(net);
        debug_str(net->bdebug, "\npush backward\n");
        backward_network(net);
        for (int i = 0; i < net->n; ++i){
            Layer *l = &net->layers[i];
            full_list_with_float(l->delta, 0, l->inputs, 1, 0);
        }
        offset += net->batch;
        if (offset >= net->num) offset -= net->num;
        n += 1;
        if (n == x){
            save_weights(net, "./data/w.weights");
            break;
        }
    }
}

void forward_network(Network *net)
{
    for (int i = 0; i < net->n; ++i){
        Layer *l = &net->layers[i];
        l->input = net->output;
        l->forward(l[0], net[0]);
        net->output = l->output;
        full_list_with_float(net->workspace, 0, net->workspace_size, 1, 0);
    }
}

void backward_network(Network *net)
{
    net->delta = NULL;
    for (int i = net->n-1; i >= 0; --i){
        Layer *l = &net->layers[i];
        l->backward(l[0], net[0]);
        net->delta = l->delta;
        full_list_with_float(net->workspace, 0, net->workspace_size, 1, 0);
        full_list_with_float(l->output, 0, l->outputs, 1, 0);
    }
}