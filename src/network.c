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
    } else if (0 == strcmp(p->type, "mse")){
        layer = make_mse_layer(p, net->batch, h, w, c);
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
        printf("%d\n", n);
        load_train_data(net, offset);
        forward_network(net);
        backward_network(net);
        for (int i = 0; i < net->n; ++i){
            Layer *l = &net->layers[i];
            fill_cpu(l->delta, l->inputs, 0, 1);
        }
        offset += net->batch;
        if (offset >= net->num) offset -= net->num;
        n += 1;
        if (n == x) {
            save_weights(net, "./backup/weights.w");
            break;
	    }
    }
}

void test(Network *net, char *test_png, char *test_label)
{
    int *w = malloc(sizeof(int));
    int *h = malloc(sizeof(int));
    int *c = malloc(sizeof(int));
    float *im = load_image_data(test_png, w, h, c);
    resize_im(im, h[0], w[0], c[0], net->height, net->width, net->input);
    net->batch = 1;
    net->output = net->input;
    net->labels[0] = get_labels(test_label)[0];
    free(w);
    free(h);
    free(c);
    free(im);
    forward_network(net);
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
